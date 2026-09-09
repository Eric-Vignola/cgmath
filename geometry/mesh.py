from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any, List, Optional, Tuple, Union

import numpy as np

# usd
try:
    from cgmath.formats.usd.prim import iter_prims
    from cgmath.formats.usd.stage import open_stage
except ImportError:
    iter_prims = None
    open_stage = None


# opencv
try:
    import cv2
except ImportError:
    cv2 = None

# scikit-image
try:
    import skimage
except ImportError:
    skimage = None

# PIL
try:
    from PIL import Image, ImageOps
except ImportError:
    Image    = None
    ImageOps = None

# trimesh
try:
    import trimesh
except ImportError:
    trimesh = None

# autodesk fbx sdk
try:
    import fbx
except ImportError:
    fbx = None


from cgmath.geometry import Data, DataList, ImmutableArray as numpy_array
from cgmath.geometry._saddle_surface import integrate, sample
from cgmath.geometry.utils import (
    average_points,
    bilinear_vectors,
    blur,
    build_tri_expand_map,
    build_uv_to_normal_map,
    compare_normals,
    composite_sprite,
    composite_sprite_aa,
    compute_centroids,
    compute_e2v_e2f,
    compute_neighbor_distances,
    compute_neighbors,
    compute_topological_neighborhood,
    compute_v2x,
    indices_replace,
    matrix_row_combine,
    matrix_row_overlaps,
    matrix_to_stream,
    point_row_overlaps,
    pxr,
    quad_match_greedy,
    rebuild_indices,
    shared_edges_test,
    split_at_hard_edges,
    stream_to_matrix,
    subdivide_catmull_clark,
)
from scipy.ndimage import convolve
from scipy.spatial import cKDTree

MESH_PRIM_TYPE = "Mesh"
UV_ATTR_PREFIX = "primvars:st"

EPSILON = np.finfo(np.float32).eps


class AreaMethod(Enum):
    ADAPTIVE    = "adaptive"    # adaptive refinement (auto-converges to tolerance)
    QUADRATURE  = "quadrature"  # n*n midpoint rule (configurable precision)
    GAUSS       = "gauss"       # 2x2 Gaussian quadrature (4 samples, fast)
    PLANAR      = "planar"
    TRIANGULATE = "triangulate"


class SampleMethod(Enum):
    BILINEAR = "bilinear"  # flat bilinear patches (default)
    BEZIER   = "bezier"    # PN Quad bicubic Bezier patches (uses vertex normals)


class TriangulateMethod(Enum):
    FAST    = "fast"     # maya like
    TRINITY = "trinity"  # mirrored shortest face area
    EVEN    = "even"     # edge 0-2
    ODD     = "odd"      # edge 1-3


@dataclass
class TriangulateRules:
    """Bundled triangulation rules for tri, quad, and n-gon faces.

    Attributes:
        rules: Per-face int array -- ``-1`` = already a triangle,
            ``0`` = split edge 0-2, ``1`` = split edge 1-3 (quads only).
        ngon_tris: Mapping of n-gon face index -> ``(T, 3)`` int array of
            **local** vertex indices (0 to ``n-1`` within the face).  Local
            indices make the triangulation topology-only and valid for any
            data sharing the same face layout (positions, UVs, etc.).
            Empty ``{}`` when no n-gons exist.
    """

    rules:     np.ndarray
    ngon_tris: dict


class RenderMethod(Enum):
    CV2     = "cv2"
    SKIMAGE = "skimage"


class Axis(Enum):
    X = "x"
    Y = "y"
    Z = "z"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            lower_value = value.lower()
            for member in cls:
                if member.value == lower_value:
                    return member
        return None


def _require_no_ngons(method_name: str, counts: np.ndarray) -> None:
    """Raise if mesh contains n-gon faces (count > 4)."""
    if np.any(counts > 4):
        raise NotImplementedError(
            f"{method_name}() does not support n-gon faces (count > 4). "
            f"Triangulate first via get_triangulate_rules() + triangulate()."
        )


def _newell_normals(
    points:    np.ndarray,
    indices:   np.ndarray,
    counts:    np.ndarray,
    normalize: bool       = True,
) -> np.ndarray:
    """Compute face normals via vectorized Newell's method.

    Works uniformly for triangles, quads, and n-gons. Uses
    ``np.bincount`` for efficient per-face accumulation without loops.

    Newell's method computes the normal of a planar polygon by summing
    cross-product contributions from consecutive edge pairs::

        N_x += (y_i - y_{i+1}) * (z_i + z_{i+1})
        N_y += (z_i - z_{i+1}) * (x_i + x_{i+1})
        N_z += (x_i - x_{i+1}) * (y_i + y_{i+1})

    Args:
        points: ``(V, 3)`` vertex positions.
        indices: Flat face-vertex index stream (length = sum of counts).
        counts: Per-face vertex count array ``(F,)``.
        normalize: If True (default), return unit normals.

    Returns:
        ``(F, 3)`` face normal array.
    """
    n_faces = counts.size

    # face boundary offsets
    offsets    = np.empty(n_faces + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    # face id for each vertex in the flat stream
    face_ids = np.repeat(np.arange(n_faces, dtype=np.intp), counts)

    # next vertex index in stream, wrapping at face boundaries
    next_idx                  = np.arange(indices.size, dtype=np.intp) + 1
    next_idx[offsets[1:] - 1] = offsets[:-1]

    # gather current and next vertex positions
    curr = points[indices]
    nxt  = points[indices[next_idx]]

    # Newell cross-product terms
    dx = curr[:, 0] - nxt[:, 0]
    dy = curr[:, 1] - nxt[:, 1]
    dz = curr[:, 2] - nxt[:, 2]
    sx = curr[:, 0] + nxt[:, 0]
    sy = curr[:, 1] + nxt[:, 1]
    sz = curr[:, 2] + nxt[:, 2]

    # accumulate per face via bincount (buffered, fast)
    normals       = np.empty((n_faces, 3), dtype=np.float64)
    normals[:, 0] = np.bincount(face_ids, weights=dy * sz, minlength=n_faces)
    normals[:, 1] = np.bincount(face_ids, weights=dz * sx, minlength=n_faces)
    normals[:, 2] = np.bincount(face_ids, weights=dx * sy, minlength=n_faces)

    if normalize:
        magnitudes                                         = np.sqrt(np.einsum("ij,ij->i", normals, normals))
        magnitudes[magnitudes < np.finfo(np.float64).tiny] = 1.0
        normals /= magnitudes[:, None]

    return normals


@dataclass(repr=False, eq=False)
class MeshData(Data):
    """
    Polygonal descriptor representing an object with edges, faces and vertices.
    """

    EQUALITY_TEST_IGNORE = ["name"]

    indices:        np.ndarray
    counts:         np.ndarray
    points:         np.ndarray
    name:           Optional[str] = None
    matrix:         Optional[np.ndarray] = numpy_array(np.eye(4))
    hole_faces:     Optional[np.ndarray] = None
    hole_counts:    Optional[np.ndarray] = None
    hole_indices:   Optional[np.ndarray] = None
    normals:        Optional[np.ndarray] = None
    normal_indices: Optional[np.ndarray] = None

    # --- cached attributes --- #
    _f2v = None  # faces to vertices
    _v2f = None  # vertices to faces
    _f2e = None  # faces to edges
    _v2e = None  # vertices to edges
    _e2f = None  # edges to faces
    _e2v = None  # edges to vertices

    _ue2f              = None  # unique edges to faces
    _ue2v              = None  # unique edges to vertices
    _f2ue              = None  # faces to unique edges
    _v2ue              = None  # vertices to unique edges
    _unique_edge_pairs = None  # unique edge pairs
    _unique_edges      = None  # unique edges

    _non_manifold_vertices = None  # non-manifold vertices
    _lamina_faces          = None  # lamina faces

    _border_vertices    = None  # vertex indices on an opened border
    _border_edges       = None  # edge indices on an opened border
    _border_faces       = None  # face indices on an opened border
    _overlap_vertices   = None  # overlapping vertex indices

    _face_normals       = None  # face normals
    _face_vertex_angles = None  # face-vertex angles

    _edge_lengths = None  # edge lengths
    _edge_normals = None  # edge normals

    _triangles        = None  # faces identified as triangles
    _quads            = None  # faces identified as quads
    _ngons            = None  # faces identified as ngons (tsk tsk)
    _degenerate_faces = None  # faces too short, or repeating a vertex
    _valence          = None  # valence per vertex
    _valid            = None  # topology validity

    # neighbors
    _v2ev = None  # vert to edge vertices
    _v2fv = None  # vert to face vertices
    _e2ve = None  # edge to vert edges
    _e2fe = None  # edge to face vertices
    _f2vf = None  # face to vert faces
    _f2ef = None  # face to edge faces

    # shells
    _shell_points = None  # shell points indices
    _shell_faces  = None  # shell faces indices
    _shell_edges  = None  # shell edges indices

    # raycast acceleration (lazily built BVH per topology)
    _bvh_bilinear = None  # BVH over per-face vertex AABBs (for bilinear raycast)
    _bvh_bezier   = None  # BVH over PN-Quad bicubic Bezier control-point AABBs

    def _compute_edges(self):
        # compute e2v and e2f
        self._ue2v, self._ue2f = compute_e2v_e2f(self.indices, self.counts)

        # derive f2e from e2f
        self._f2ue = stream_to_matrix(np.arange(self._ue2f.size), self.counts)

        # derive e2f from e2v
        edges, e0, e1 = matrix_row_overlaps(self._ue2v, exact=False)
        self._unique_edges      = edges
        self._unique_edge_pairs = np.array([e0, e1]).T

        self._ue2f = matrix_row_combine(self._ue2f, e0, e1)
        self._ue2f = matrix_row_combine(self._ue2f, e1, e0)
        self._e2f  = self._ue2f[edges]
        self._e2v  = self._ue2v[edges]
        self._f2e  = indices_replace(self._f2ue, e0, e1, collapse=True)

    @property
    def area(self):
        return np.sum(self.get_face_areas())

    @property
    def f2v(self):
        # face to vertices
        if self._f2v is None:
            self._f2v = stream_to_matrix(self.indices, self.counts)
        return self._f2v

    @property
    def v2f(self):
        # vertices to faces
        if self._v2f is None:
            unique, unique_counts = np.unique(self.indices, return_counts=True)
            self._v2f = compute_v2x(
                self.f2v, (unique[unique >= 0].size, unique_counts.max())
            )
        return self._v2f

    @property
    def f2e(self):
        # faces to edges
        if self._f2e is None:
            self._compute_edges()
        return self._f2e

    @property
    def f2ue(self):
        # faces to unique edges
        if self._f2ue is None:
            self._compute_edges()
        return self._f2ue

    @property
    def ue2f(self):
        # unique edges to faces
        if self._ue2f is None:
            self._compute_edges()
        return self._ue2f

    @property
    def ue2v(self):
        # unique edges to vertices
        if self._ue2v is None:
            self._compute_edges()
        return self._ue2v

    @property
    def ue(self):
        # unique edge indices
        if self._unique_edges is None:
            self._compute_edges()
        return self._unique_edges

    @property
    def uep(self):
        # unique edge index pairs
        if self._unique_edge_pairs is None:
            self._compute_edges()
        return self._unique_edge_pairs

    @property
    def v2e(self):
        # vertices to edges
        if self._v2e is None:
            unique, unique_counts = np.unique(self.e2v, return_counts=True)
            self._v2e = compute_v2x(
                self.e2v, (unique[unique >= 0].size, unique_counts.max())
            )
        return self._v2e

    @property
    def v2ue(self):
        # vertices to unique edges
        if self._v2ue is None:
            unique, unique_counts = np.unique(self.ue2v, return_counts=True)
            self._v2ue = compute_v2x(
                self.ue2v, (unique[unique >= 0].size, unique_counts.max())
            )
        return self._v2ue

    @property
    def e2f(self):
        # edges to faces
        if self._e2f is None:
            self._compute_edges()
        return self._e2f

    @property
    def e2v(self):
        # edges to vertices
        if self._e2v is None:
            self._compute_edges()
        return self._e2v

    def faces_to_vertices(self, faces: np.ndarray) -> np.ndarray:
        """returns the vertices for a given set of faces"""
        indices = np.unique(self.f2v[faces])
        return indices[indices >= 0]

    def edges_to_vertices(self, edges: np.ndarray) -> np.ndarray:
        """returns the vertices for a given set of edges"""
        return np.unique(self.e2v[edges])

    def edges_to_vertex_pairs(self, edges: np.ndarray) -> np.ndarray:
        """returns the vertex pairs for a given set of edges"""

        # return vertices sorted low-high
        pairs = np.sort(self.e2v[edges], axis=1)
        return pairs.ravel()

    def vertices_to_faces(self, verts: np.ndarray) -> np.ndarray:
        """returns the vertices for a given set of faces"""
        indices = np.unique(self.v2f[verts])
        return indices[indices >= 0]

    def vertices_to_edges(self, verts: np.ndarray) -> np.ndarray:
        """returns the edges for a given set of vertices"""

        edges = np.unique(self.v2e[verts])
        return edges[edges >= 0]

    def vertex_pairs_to_edges(self, verts: np.ndarray) -> np.ndarray:
        """returns the edges for a given set of vertex pairs"""

        verts = np.asarray(verts, dtype="int")
        verts = verts.reshape(-1, 2)       # reshape [0,1,1,2] --> [[0,1],[1,2]]
        verts = np.sort(verts, axis=1)     # sort low-high (in case unsorted)
        edges = np.sort(self.e2v, axis=1)  # sort low-high

        # find the matching pairs
        tree = cKDTree(edges)
        d, idx = tree.query(verts)

        # pairs which are not found will have a distance > 0
        test = np.where(d > 0)[0]
        if test.size > 0:
            raise RuntimeError(f"{repr(verts[test])} are not valid vertex pairs!")

        return idx

    @property
    def valid(self):
        # validity test:
        # all indices are used and within range (0 to npoints-1)
        # all polygon counts sum up to point size
        # every face is a polygon, or absent -- a count of zero is a UVData
        # hole and stays legal
        if self._valid is None:
            if self.indices.size == 0:
                self._valid = self.points.shape[0] == 0
            else:
                unique_idx = np.unique(self.indices)
                self._valid = np.all(
                    [
                        unique_idx[0] == 0,
                        unique_idx[-1] == self.points.shape[0] - 1,
                        unique_idx.size == self.points.shape[0],
                        np.sum(self.counts) == self.indices.size,
                        self.get_degenerate_faces().size == 0,
                    ]
                )
        return self._valid

    @property
    def triangles(self):
        """returns the triangle count for the mesh"""
        return self.get_triangles().size

    @property
    def quads(self):
        """returns the quad count for the mesh"""
        return self.get_quads().size

    @property
    def ngons(self):
        """returns the ngon count for the mesh"""
        return self.get_ngons().size

    @property
    def has_degenerate_faces(self):
        """returns True if any face cannot be a polygon"""
        return self.get_degenerate_faces().size > 0

    @property
    def valence(self):
        """returns the max valence for the mesh vertices"""
        return self.get_valence().max()

    @property
    def point_count(self):
        """returns the mesh's vertex count"""
        return self.points.shape[0]

    @property
    def face_count(self):
        """returns the mesh's face count"""
        return self.counts.size

    @property
    def edge_count(self):
        """returns the mesh's face count"""
        return self.e2v.shape[0]

    @property
    def open(self):
        """returns True if mesh has border edges"""
        return len(self.get_border_vertices()) > 0

    @property
    def closed(self):
        """returns True if mesh has no border edges"""
        return not self.open

    @property
    def geometry(self):
        """returns what you'd consider the geometric construct"""
        return self.f2v

    @property
    def points4(self):
        """returns self.points as a 4d sequence for matrix math purposes"""
        points                            = np.zeros((self.points.shape[0], 4))
        points[:, : self.points.shape[1]] = self.points
        points[:, -1]                     = 1
        return points

    def get_triangles(self):
        """returns the face indices that are triangles"""
        if self._triangles is None:
            self._triangles = np.where(self.counts == 3)[0]
        return self._triangles

    def get_quads(self):
        """returns the face indices that are quads"""
        if self._quads is None:
            self._quads = np.where(self.counts == 4)[0]
        return self._quads

    def get_ngons(self):
        """returns the face ngon count"""
        if self._ngons is None:
            self._ngons = np.where(self.counts > 4)[0]
        return self._ngons

    def get_degenerate_faces(self):
        """returns the face indices that cannot be a polygon

        A face qualifies by holding fewer than three vertices, or by naming
        the same vertex more than once -- a triangle written as a quad by
        repeating a vertex still encloses area, so get_zero_area_faces will
        not see it. A count of zero is a hole, which UVData uses
        deliberately and fill_holes repairs, so it is not reported here.
        """
        if self._degenerate_faces is None:
            geometry = self.geometry
            sentinel = self.points.shape[0]
            ordered  = np.sort(np.where(geometry >= 0, geometry, sentinel), axis=1)

            repeated = np.zeros(self.counts.shape, dtype=bool)
            if ordered.shape[1] > 1:
                repeated = (
                    (np.diff(ordered, axis=1) == 0) & (ordered[:, :-1] < sentinel)
                ).any(axis=1)

            self._degenerate_faces = np.where(
                (self.counts > 0) & ((self.counts < 3) | repeated)
            )[0]
        return self._degenerate_faces

    def get_valence(self):
        """returns the number of edges connected to a vertex"""
        if self._valence is None:
            self._valence = np.sum(self.v2e >= 0, axis=1)
        return self._valence

    # ------------------------------ GEOMETRY TESTS ------------------------------ #

    def get_unused_points(self):
        """returns indices of unused points"""
        unique_idx = np.unique(self.indices)
        return np.setdiff1d(np.arange(self.points.shape[0]), unique_idx)

    def get_zero_area_faces(self, tolerance=1e-12):
        """returns the face indices enclosing no area

        Catches what a vertex count cannot: collinear points and coincident
        points. Note that a face repeating a vertex is only caught here when
        the repeat cancels, so get_degenerate_faces covers that case.

        Args:
            tolerance: Faces at or below this area count as empty.
        """
        faces = np.where(self.counts > 0)[0]
        if faces.size == 0:
            return np.array([], dtype=int)

        points = self.points
        if points.shape[1] < 3:
            # newell needs three components, and a uv plane's shoelace area
            # is exactly the term the padding leaves behind
            padded                       = np.zeros((points.shape[0], 3))
            padded[:, : points.shape[1]] = points
            points                       = padded

        # _newell_normals miscomputes the face preceding a zero count, and a
        # hole contributes no entries to the stream, so dropping holes from
        # counts alone leaves indices already correct
        normals = _newell_normals(
            points, self.indices, self.counts[faces], normalize=False
        )
        return faces[np.linalg.norm(normals, axis=1) * 0.5 <= tolerance]

    def get_lamina_faces(self):
        """returns lamina faces"""
        if self._lamina_faces is None:
            self._lamina_faces = matrix_row_overlaps(self.f2v, exact=False)[1]

        return self._lamina_faces

    def get_non_manifold_vertices(self):
        """returns non-manifold vertices"""

        if self._non_manifold_vertices is None:
            # winding order test
            a    = self.ue2v[self.uep[:, 0]]
            b    = self.ue2v[self.uep[:, 1]]
            mask = np.where((a == b).all(axis=1))[0]
            winding_order_test = np.concatenate(
                [self.ue2v[self.uep[:, 0][mask]], self.ue2v[self.uep[:, 1][mask]]]
            )

            # shared edge test
            mask             = np.sum(self.e2f >= 0, axis=1) > 2
            shared_edge_test = self.e2v[mask]

            # shared vertex test
            indices = np.where(np.sum(self.v2f >= 0, axis=1) > 1)[0]
            edges   = self.f2e[self.v2f[indices]]
            if edges.size > 0:
                edges              = edges.reshape(edges.shape[0], -1)
                shared_vertex_test = indices[shared_edges_test(edges)]
            else:
                shared_vertex_test = []

            # disconnected fan test
            disconnected_fan_test = np.array([], dtype="int")
            for border in self.get_border_edges():
                vertices = self.e2v[border]
                vertices, counts = np.unique(vertices, return_counts=True)
                vertices = vertices[counts > 2]
                disconnected_fan_test = np.concatenate(
                    [disconnected_fan_test, vertices]
                )

            self._non_manifold_vertices = np.concatenate(
                [
                    winding_order_test.ravel(),
                    shared_edge_test.ravel(),
                    shared_vertex_test,
                    disconnected_fan_test,
                ]
            )

            self._non_manifold_vertices = np.unique(self._non_manifold_vertices)

        return self._non_manifold_vertices

    # -------------------------------- NEIGHBORS --------------------------------- #
    def get_face_vertex_geodesic_neighborhood(
        self,
        n:              int        | None = None,
        max_distance:   float      | None = None,
        indices:        np.ndarray | None = None,
        distance_format                   = "sparse",
    ):
        """
        Compute the topological face vertex neighborhood with distances using Dijkstra's algorithm

        Args:
            n: Number of hops to grow the neighborhood. If None, explores all reachable vertices.
            max_distance: Maximum cumulative distance threshold. If None, no distance limit.
            indices: Vertex indices to compute neighborhoods for. If None, computes for all vertices.
            distance_format: Output format for distances. "sparse", "dense", or "shortest".
        """

        # get connectivity matrix and sample distances
        conn_matrix    = self.get_face_vertex_neighbors()
        conn_distances = compute_neighbor_distances(conn_matrix, self.points)

        # return the desired data
        return compute_topological_neighborhood(
            conn_matrix,
            num_hops        = n,
            indices         = indices,
            distances       = conn_distances,
            max_distance    = max_distance,
            distance_format = distance_format,
        )

    def get_edge_vertex_geodesic_neighborhood(
        self,
        n:              int        | None = None,
        max_distance:   float      | None = None,
        indices:        np.ndarray | None = None,
        distance_format                   = "sparse",
    ):
        """
        Compute the topological edge vertex neighborhood with distances using Dijkstra's algorithm

        Args:
            n: Number of hops to grow the neighborhood. If None, explores all reachable vertices.
            max_distance: Maximum cumulative distance threshold. If None, no distance limit.
            indices: Vertex indices to compute neighborhoods for. If None, computes for all vertices.
            distance_format: Output format for distances. "sparse", "dense", or "shortest".
        """
        # get connectivity matrix and sample distances
        conn_matrix    = self.get_edge_vertex_neighbors()
        conn_distances = compute_neighbor_distances(conn_matrix, self.points)

        # return the desired data
        return compute_topological_neighborhood(
            conn_matrix,
            num_hops        = n,
            indices         = indices,
            distances       = conn_distances,
            max_distance    = max_distance,
            distance_format = distance_format,
        )

    def get_face_edge_geodesic_neighborhood(
        self,
        n:              int        | None = None,
        max_distance:   float      | None = None,
        indices:        np.ndarray | None = None,
        distance_format                   = "sparse",
    ):
        """
        Compute the topological face edge neighborhood with distances using Dijkstra's algorithm

        Args:
            n: Number of hops to grow the neighborhood. If None, explores all reachable edges.
            max_distance: Maximum cumulative distance threshold. If None, no distance limit.
            indices: Edge indices to compute neighborhoods for. If None, computes for all edges.
            distance_format: Output format for distances. "sparse", "dense", or "shortest".
        """
        # get connectivity matrix and sample distances
        conn_matrix    = self.get_face_edge_neighbors()
        edge_centers   = self.points[self.e2v].mean(axis=1)
        conn_distances = compute_neighbor_distances(conn_matrix, edge_centers)

        # return the desired data
        return compute_topological_neighborhood(
            conn_matrix,
            num_hops        = n,
            indices         = indices,
            distances       = conn_distances,
            max_distance    = max_distance,
            distance_format = distance_format,
        )

    def get_vertex_edge_geodesic_neighborhood(
        self,
        n:              int        | None = None,
        max_distance:   float      | None = None,
        indices:        np.ndarray | None = None,
        distance_format                   = "sparse",
    ):
        """
        Compute the topological vertex edge neighborhood with distances using Dijkstra's algorithm

        Args:
            n: Number of hops to grow the neighborhood. If None, explores all reachable edges.
            max_distance: Maximum cumulative distance threshold. If None, no distance limit.
            indices: Edge indices to compute neighborhoods for. If None, computes for all edges.
            distance_format: Output format for distances. "sparse", "dense", or "shortest".
        """
        # get connectivity matrix and sample distances
        conn_matrix    = self.get_vertex_edge_neighbors()
        edge_centers   = self.points[self.e2v].mean(axis=1)
        conn_distances = compute_neighbor_distances(conn_matrix, edge_centers)

        # return the desired data
        return compute_topological_neighborhood(
            conn_matrix,
            num_hops        = n,
            indices         = indices,
            distances       = conn_distances,
            max_distance    = max_distance,
            distance_format = distance_format,
        )

    def get_vertex_face_geodesic_neighborhood(
        self,
        n:              int        | None = None,
        max_distance:   float      | None = None,
        indices:        np.ndarray | None = None,
        distance_format                   = "sparse",
    ):
        """
        Compute the topological vertex face neighborhood with distances using Dijkstra's algorithm

        Args:
            n: Number of hops to grow the neighborhood. If None, explores all reachable faces.
            max_distance: Maximum cumulative distance threshold. If None, no distance limit.
            indices: Face indices to compute neighborhoods for. If None, computes for all faces.
            distance_format: Output format for distances. "sparse", "dense", or "shortest".
        """
        # get connectivity matrix and sample distances
        conn_matrix    = self.get_vertex_face_neighbors()
        face_centroids = compute_centroids(self.points, self.f2v)
        conn_distances = compute_neighbor_distances(conn_matrix, face_centroids)

        # return the desired data
        return compute_topological_neighborhood(
            conn_matrix,
            num_hops        = n,
            indices         = indices,
            distances       = conn_distances,
            max_distance    = max_distance,
            distance_format = distance_format,
        )

    def get_edge_face_geodesic_neighborhood(
        self,
        n:              int        | None = None,
        max_distance:   float      | None = None,
        indices:        np.ndarray | None = None,
        distance_format                   = "sparse",
    ):
        """
        Compute the topological edge face neighborhood with distances using Dijkstra's algorithm

        Args:
            n: Number of hops to grow the neighborhood. If None, explores all reachable faces.
            max_distance: Maximum cumulative distance threshold. If None, no distance limit.
            indices: Face indices to compute neighborhoods for. If None, computes for all faces.
            distance_format: Output format for distances. "sparse", "dense", or "shortest".
        """
        # get connectivity matrix and sample distances
        conn_matrix    = self.get_edge_face_neighbors()
        face_centroids = compute_centroids(self.points, self.f2v)
        conn_distances = compute_neighbor_distances(conn_matrix, face_centroids)

        # return the desired data
        return compute_topological_neighborhood(
            conn_matrix,
            num_hops        = n,
            indices         = indices,
            distances       = conn_distances,
            max_distance    = max_distance,
            distance_format = distance_format,
        )

    def get_edge_vertex_neighbors(self, n: int = 0) -> np.ndarray:
        """return a matrix of vertices neighboring connected edges to vertices"""
        if n == 0:
            if self._v2ev is None:
                self._v2ev = compute_neighbors(self.v2e, self.e2v, n=n)
            return self._v2ev
        return compute_neighbors(self.v2e, self.e2v, n=n)

    def get_face_vertex_neighbors(self, n: int = 0) -> np.ndarray:
        """return a matrix of vertices neighboring connected faces to vertices"""
        if n == 0:
            if self._v2fv is None:
                self._v2fv = compute_neighbors(self.v2f, self.f2v, n=n)
            return self._v2fv
        return compute_neighbors(self.v2f, self.f2v, n=n)

    def get_vertex_edge_neighbors(self, n: int = 0) -> np.ndarray:
        """return a matrix of edges neighboring connected vertices to edges"""
        if n == 0:
            if self._e2ve is None:
                self._e2ve = compute_neighbors(self.e2v, self.v2e, n=n)
            return self._e2ve
        return compute_neighbors(self.e2v, self.v2e, n=n)

    def get_face_edge_neighbors(self, n: int = 0) -> np.ndarray:
        """return a matrix of edges neighboring connected faces to edges"""
        if n == 0:
            if self._e2fe is None:
                self._e2fe = compute_neighbors(self.e2f, self.f2e, n=n)
            return self._e2fe
        return compute_neighbors(self.e2f, self.f2e, n=n)

    def get_vertex_face_neighbors(self, n: int = 0) -> np.ndarray:
        """return a matrix of faces neighboring connected vertices to faces"""
        if n == 0:
            if self._f2vf is None:
                self._f2vf = compute_neighbors(self.f2v, self.v2f, n=n)
            return self._f2vf
        return compute_neighbors(self.f2v, self.v2f, n=n)

    def get_edge_face_neighbors(self, n: int = 0) -> np.ndarray:
        """return a matrix of faces neighboring connected edges to faces"""
        if n == 0:
            if self._f2ef is None:
                self._f2ef = compute_neighbors(self.f2e, self.e2f, n=n)
            return self._f2ef
        return compute_neighbors(self.f2e, self.e2f, n=n)

    def _shrink_indices(
        self, indices: np.ndarray, neighbors: np.ndarray, borders: np.ndarray, n=1
    ):
        """shrinks indices by n steps"""
        indices = np.asarray(indices, dtype=int)

        while n > 0 and indices.size > 0:
            n -= 1
            expanded = neighbors[indices]
            mask     = np.isin(expanded, indices) | (expanded == -1)
            indices  = indices[np.all(mask, axis=1)]

            # remove borders (happens only once)
            if borders is not None:
                indices = np.setdiff1d(indices, borders)
                borders = None

        return indices

    def shrink_face_vertex_indices(self, indices: np.ndarray, n: int = 1) -> np.ndarray:
        """shrinks face vertex indices by n steps"""
        neighbors = self.get_face_vertex_neighbors()
        borders   = self.get_border_vertices(flatten=True)
        return self._shrink_indices(indices, neighbors, borders, n=n)

    def shrink_edge_vertex_indices(self, indices: np.ndarray, n: int = 1) -> np.ndarray:
        """shrinks edge vertex indices by n steps"""
        neighbors = self.get_edge_vertex_neighbors()
        borders   = self.get_border_vertices(flatten=True)
        return self._shrink_indices(indices, neighbors, borders, n=n)

    def shrink_face_edge_indices(self, indices: np.ndarray, n: int = 1) -> np.ndarray:
        """shrinks face edge indices by n steps"""
        neighbors = self.get_face_edge_neighbors()
        borders   = self.get_border_edges(flatten=True)
        return self._shrink_indices(indices, neighbors, borders, n=n)

    def shrink_vertex_edge_indices(self, indices: np.ndarray, n: int = 1) -> np.ndarray:
        """shrinks vertex edge indices by n steps"""
        neighbors = self.get_vertex_edge_neighbors()
        borders   = self.get_border_edges(flatten=True)
        return self._shrink_indices(indices, neighbors, borders, n=n)

    def shrink_vertex_face_indices(self, indices: np.ndarray, n: int = 1) -> np.ndarray:
        """shrinks vertex face indices by n steps"""
        neighbors = self.get_vertex_face_neighbors()
        borders   = self.get_border_faces(flatten=True)
        return self._shrink_indices(indices, neighbors, borders, n=n)

    def shrink_edge_face_indices(self, indices: np.ndarray, n: int = 1) -> np.ndarray:
        """shrinks vertex face indices by n steps"""
        neighbors = self.get_edge_face_neighbors()
        borders   = self.get_border_faces(flatten=True)
        return self._shrink_indices(indices, neighbors, borders, n=n)

    def _grow_indices(self, indices: np.ndarray, neighbors: np.ndarray, n=1):
        """grows indices by n steps"""
        indices = np.asarray(indices, dtype=int)
        while n > 0 and indices.size > 0 and neighbors.size > neighbors.shape[0]:
            n -= 1
            grown   = np.concatenate([indices, neighbors[indices].ravel()])
            grown   = grown[grown >= 0]  # remove -1's
            indices = np.unique(grown)

        return indices

    def grow_face_vertex_indices(self, indices: np.ndarray, n: int = 1) -> np.ndarray:
        """grows face vertex indices by n steps"""
        neighbors = self.get_face_vertex_neighbors()
        return self._grow_indices(indices, neighbors, n=n)

    def grow_edge_vertex_indices(self, indices: np.ndarray, n: int = 1) -> np.ndarray:
        """grows edge vertex indices by n steps"""
        neighbors = self.get_edge_vertex_neighbors()
        return self._grow_indices(indices, neighbors, n=n)

    def grow_vertex_edge_indices(self, indices: np.ndarray, n: int = 1) -> np.ndarray:
        """grows vertex edge indices by n steps"""
        neighbors = self.get_vertex_edge_neighbors()
        return self._grow_indices(indices, neighbors, n=n)

    def grow_face_edge_indices(self, indices: np.ndarray, n: int = 1) -> np.ndarray:
        """grows face edge indices by n steps"""
        neighbors = self.get_face_edge_neighbors()
        return self._grow_indices(indices, neighbors, n=n)

    def grow_edge_face_indices(self, indices: np.ndarray, n: int = 1) -> np.ndarray:
        """grows edge face indices by n steps"""
        neighbors = self.get_edge_face_neighbors()
        return self._grow_indices(indices, neighbors, n=n)

    def grow_vertex_face_indices(self, indices: np.ndarray, n: int = 1) -> np.ndarray:
        """grows vertex face indices by n steps"""
        neighbors = self.get_vertex_face_neighbors()
        return self._grow_indices(indices, neighbors, n=n)

    # --------------------------------- CLUSTERS --------------------------------- #
    @staticmethod
    def _compute_clusters(
        indices: np.ndarray, neighbors: np.ndarray
    ) -> List[np.ndarray]:
        def dfs(node, component):
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)

                    # add neighbors that are in the cluster and not visited
                    for neighbor in neighbors[current]:
                        if (
                            neighbor > -1
                            and neighbor in cluster_set
                            and neighbor not in visited
                        ):
                            stack.append(neighbor)

        # find all connected indices
        cluster     = np.unique(indices)
        cluster     = cluster[cluster > -1]  # remove -1's
        cluster_set = set(cluster)
        visited     = set()
        components  = []
        for node in cluster:
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(np.sort(component))

        return components

    def get_face_vertex_clusters(self, indices: np.ndarray) -> List[np.ndarray]:
        """convert face vertex indices to clusters"""
        neighbors = self.get_face_vertex_neighbors()
        return self._compute_clusters(indices, neighbors)

    def get_edge_vertex_clusters(self, indices: np.ndarray) -> List[np.ndarray]:
        """convert edge vertex indices to clusters"""
        neighbors = self.get_edge_vertex_neighbors()
        return self._compute_clusters(indices, neighbors)

    def get_vertex_edge_clusters(self, indices: np.ndarray) -> List[np.ndarray]:
        """convert vertex edge indices to clusters"""
        neighbors = self.get_vertex_edge_neighbors()
        return self._compute_clusters(indices, neighbors)

    def get_face_edge_clusters(self, indices: np.ndarray) -> List[np.ndarray]:
        """convert face edge indices to clusters"""
        neighbors = self.get_face_edge_neighbors()
        return self._compute_clusters(indices, neighbors)

    def get_edge_face_clusters(self, indices: np.ndarray) -> List[np.ndarray]:
        """convert edge face indices to clusters"""
        neighbors = self.get_edge_face_neighbors()
        return self._compute_clusters(indices, neighbors)

    def get_vertex_face_clusters(self, indices: np.ndarray) -> List[np.ndarray]:
        """convert vertex face indices to clusters"""
        neighbors = self.get_vertex_face_neighbors()
        return self._compute_clusters(indices, neighbors)

    @property
    def shell_edges(self):
        """returns the shells as edge indices"""
        if self._shell_edges is None:
            indices           = np.arange(self.edge_count)
            self._shell_edges = self.get_face_edge_clusters(indices)

        return self._shell_edges

    @property
    def shell_points(self):
        """returns the shells as point indices"""
        if self._shell_points is None:
            indices            = np.arange(self.point_count)
            self._shell_points = self.get_face_vertex_clusters(indices)

        return self._shell_points

    @property
    def shell_faces(self):
        """returns the shells as face indices"""
        if self._shell_faces is None:
            indices           = np.arange(self.face_count)
            self._shell_faces = self.get_vertex_face_clusters(indices)

        return self._shell_faces

    @property
    def shell_count(self):
        """returns the number of shells"""
        return len(self.shell_faces)

    # --------------------------------- BORDERS ---------------------------------- #

    def _compute_borders(self):
        """computes polygon borders"""
        if self._e2v is None:
            self._compute_edges()

        # build border data
        counts       = np.sum(self.e2f >= 0, axis=1) == 1
        edge_indices = np.where(counts)[0]
        edge_borders = self.e2v[edge_indices]

        # start marching
        self._border_vertices = []
        self._border_edges    = []

        while edge_borders.shape[0] > 0:
            vertices     = edge_borders[0]
            edges        = [edge_indices[0]]
            edge_borders = edge_borders[1:]
            edge_indices = edge_indices[1:]

            while edge_borders.shape[0] > 0:
                mask      = np.isin(edge_borders, vertices)
                mask_rows = np.sum(mask, axis=1) > 0
                if np.any(mask_rows):
                    new_vertices = edge_borders[mask_rows][~mask[mask_rows]]
                    new_edges    = edge_indices[mask_rows]
                    vertices     = np.unique(np.concatenate((new_vertices, vertices)))
                    edges        = np.unique(np.concatenate((new_edges, edges)))
                else:
                    break

                edge_borders = edge_borders[~mask_rows]
                edge_indices = edge_indices[~mask_rows]

            # append to loops
            self._border_vertices.append(vertices)
            self._border_edges.append(edges)

            # sort by size
            self._border_edges    = sorted(self._border_edges, key=len)
            self._border_vertices = sorted(self._border_vertices, key=len)

    def get_border_vertices(self, flatten=False):
        """returns all border vertices"""

        if self._border_vertices is None:
            self._compute_borders()

        if flatten and len(self._border_vertices) > 0:
            if len(self._border_vertices) == 1:
                return self._border_vertices[0]
            return np.unique(np.concatenate(self._border_vertices))

        return self._border_vertices

    def get_border_edges(self, flatten=False):
        """returns all border edges"""

        if self._border_edges is None:
            self._compute_borders()

        if flatten and len(self._border_edges) > 0:
            if len(self._border_edges) == 1:
                return self._border_edges[0]
            return np.unique(np.concatenate(self._border_edges))

        return self._border_edges

    def get_border_faces(self, flatten=False):
        """returns all border faces"""
        if self._border_faces is None:
            self._compute_borders()
            self._border_faces = []
            for vertices in self.get_border_vertices():
                faces = np.any(np.isin(self.f2v, vertices), axis=1)
                self._border_faces.append(np.where(faces)[0])

        if flatten and len(self._border_faces) > 0:
            if len(self._border_faces) == 1:
                return self._border_faces[0]
            return np.unique(np.concatenate(self._border_faces))

        return self._border_faces

    def get_overlap_vertices(self, tolerance=1e-6):
        """returns overlapping pairs of vertices"""
        if self._overlap_vertices is None:
            tree                   = cKDTree(self.points)
            duplicates             = tree.query_pairs(r=tolerance, output_type="ndarray")
            self._overlap_vertices = duplicates

        return self._overlap_vertices

    def get_face_areas(
        self,
        method:    AreaMethod | str = AreaMethod.GAUSS,
        samples:   int              = 10,
        tolerance: float            = 1e-4,
    ):
        """returns a face area approximation"""
        method = AreaMethod(method)
        if method != AreaMethod.TRIANGULATE:
            _require_no_ngons("get_face_areas", self.counts)

        if method in (AreaMethod.ADAPTIVE, AreaMethod.QUADRATURE, AreaMethod.GAUSS):
            return integrate(
                self.points,
                self.geometry,
                samples   = samples,
                method    = method.value,
                tolerance = tolerance,
            )

        elif method == AreaMethod.PLANAR:
            # split the mesh into individual faces
            indices = np.arange(self.counts.sum())
            points  = self.points[self.geometry[self.geometry >= 0]]
            planar  = MeshData(indices=indices, points=points, counts=self.counts)

            # return planar areas
            uv = np.ones((planar.face_count, 2)) * 0.5
            U, V = bilinear_vectors(planar.points, planar.geometry, uv)
            N = np.cross(U, V)
            N /= (np.einsum("...i,...i", N, N) ** 0.5)[:, None]
            N = np.repeat(N, planar.counts, axis=0)

            p = (
                planar.points[planar.geometry]
                - planar.points[planar.geometry][:, 0, None]
            )
            p             = p[planar.geometry >= 0]
            d             = np.einsum("...i,...i", p, N)
            planar.points = planar.points - N * d[:, None]

            return planar.get_face_areas(
                samples   = samples,
                method    = AreaMethod.GAUSS,
                tolerance = tolerance,
            )

        elif method == AreaMethod.TRIANGULATE:
            # return areas as triangulated faces
            copy  = self.copy()
            rules = self.get_triangulate_rules(method="fast")
            copy.triangulate(rules=rules)  # preserve_order=True keeps face order
            tri_areas = copy.get_face_areas(
                samples   = samples,
                method    = AreaMethod.GAUSS,
                tolerance = tolerance,
            )

            # map triangulated face areas back to original faces:
            # each tri produced 1 output face, each quad produced 2,
            # each n-gon produced len(ngon_tris[face_id]).
            tri_ids                = np.where(self.counts == 3)[0]
            quad_ids               = np.where(self.counts == 4)[0]
            out_per_face           = np.ones(self.counts.size, dtype=np.intp)
            out_per_face[quad_ids] = 2
            for face_id, local_tris in rules.ngon_tris.items():
                out_per_face[face_id] = len(local_tris)

            out_offsets    = np.empty(self.counts.size + 1, dtype=np.intp)
            out_offsets[0] = 0
            np.cumsum(out_per_face, out=out_offsets[1:])

            triangulated_areas          = np.zeros(self.counts.size, dtype=tri_areas.dtype)
            triangulated_areas[tri_ids] = tri_areas[out_offsets[tri_ids]]
            triangulated_areas[quad_ids] = (
                tri_areas[out_offsets[quad_ids]] + tri_areas[out_offsets[quad_ids] + 1]
            )
            for face_id, local_tris in rules.ngon_tris.items():
                start                       = out_offsets[face_id]
                n_tris                      = len(local_tris)
                triangulated_areas[face_id] = tri_areas[start : start + n_tris].sum()

            return triangulated_areas

        else:
            raise ValueError(f"Unknown area method: {method}")

    # --- normals

    def get_face_normals(self, force: bool = False) -> np.ndarray:
        """Returns the face normals. Computes it if not already computed.

        Uses vectorized Newell's method, which works uniformly for
        triangles, quads, and n-gons.

        Args:
            force: If True, force recomputes the normals.
        """
        if self._face_normals is None or force:
            self._face_normals = _newell_normals(self.points, self.indices, self.counts)
        return self._face_normals

    def get_face_vertex_angles(self, force: bool = False) -> np.ndarray:
        """Compute vertex angles in radians as a per face per vertex array
        matching `self.indices`.

        Args:
            force: If True, force recomputes the face vertex angles.
        """
        _require_no_ngons("get_face_vertex_angles", self.counts)
        if self._face_vertex_angles is not None and not force:
            return self._face_vertex_angles

        # ensure 4 max vert count per face (triangles only mesh will have 3 max vert count)
        indices                              = np.ones((self.geometry.shape[0], 4), dtype=int) * -1
        indices[:, : self.geometry.shape[1]] = self.geometry

        # consider triangles as quads with a duplicated 4th edge to keep the code vectorized
        i1 = np.where(
            self.counts[:, None] > 3, indices[:, [1, 2, 3, 0]], indices[:, [1, 2, 0, 0]]
        )
        i0 = np.where(
            self.counts[:, None] > 3, indices[:, [0, 1, 2, 3]], indices[:, [0, 1, 2, 2]]
        )
        vtx0 = self.points[i0]
        vtx1 = self.points[i1]

        # compute each edge
        e0, e1, e2, e3 = (vtx1 - vtx0).swapaxes(0, 1)

        # compute each edge length (einsum is ~10x faster than np.linalg.norm)
        with np.errstate(divide="ignore", invalid="ignore"):
            l0 = np.einsum("...i,...i", e0, e0) ** 0.5
            l1 = np.einsum("...i,...i", e1, e1) ** 0.5
            l2 = np.einsum("...i,...i", e2, e2) ** 0.5
            l3 = np.einsum("...i,...i", e3, e3) ** 0.5

            # normalize each edge
            e0 /= l0[:, None]
            e1 /= l1[:, None]
            e2 /= l2[:, None]
            e3 /= l3[:, None]

            # compute dot products (einsum is ~10x faster than np.dot)
            d0 = np.einsum("ij,ij->i", e0, 0 - e3)
            d1 = np.einsum("ij,ij->i", e1, 0 - e0)
            d2 = np.einsum("ij,ij->i", e2, 0 - e1)
            d3 = np.einsum("ij,ij->i", e3, 0 - e2)

            # compute angles and recompose geomety/angle matrix
            angles = np.array([d0, d1, d2, d3]).T
            angles = np.arccos(angles)

            # flatten to per face per vertex array, handle mix of quads/tris or just tris
            if self.counts.max() > 3:
                self._face_vertex_angles = angles[self.geometry >= 0]
            else:
                self._face_vertex_angles = angles[:, :3].ravel()

            return self._face_vertex_angles

    def get_closest_points(
        self, points: np.ndarray, k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the distances and indices of the closest points to the given points.

        Args:
            points: A list of points to compute the closest points to.
            k: the number of closest points to find for each input point.

        Returns:
            The closest distances and points to the given points.
        """
        tree = cKDTree(self.points)
        return tree.query(points, k)

    def get_vertex_normals(
        self,
        angle_weighted: bool              = True,
        area_weighted:  bool              = True,
        force:          bool              = False,
        _areas:         np.ndarray | None = None,
    ) -> np.ndarray:
        """Computes vertex normals from face normals.

        For reference, how maya calculates vertex normals:
        https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/How-Maya-computes-face-normals.html

        Maya's default behavior is: angle_weighted = True, area_weighted = True

        Args:
            angle_weighted: If True, multiply vertex-face angles to each face-vertex normal.
            area_weighted: If True, multiply face area to each face-vertex normal.
            force: If True, force recomputes all necessary data.

        Returns:
            The vertex normal array.
        """
        # copy face normals into face-vertex normals
        face_normals = self.get_face_normals(force=force)
        vert_normals = np.repeat(face_normals, self.counts, axis=0)

        # apply face area weighting
        if area_weighted:
            areas = self.get_face_areas() if _areas is None else _areas
            areas = np.repeat(areas, self.counts, axis=0)
            vert_normals *= areas[:, None]

        # apply vertex angle weighting
        if angle_weighted:
            vert_normals *= self.get_face_vertex_angles(force=force)[:, None]

        return self.sum_vertex_normals(vert_normals)

    def sum_vertex_normals(self, face_vertex_normals: np.ndarray) -> np.ndarray:
        """Given a face-vertex normal array, sums the normals at of each unique
        vertex indices. This converts face-vertex normals to vertex normals.
        """
        # sum all the vectors at the given indices
        _, inverse_indices = np.unique(self.indices, return_inverse=True)
        summed_normals = np.zeros((self.point_count, face_vertex_normals.shape[1]))
        np.add.at(summed_normals, inverse_indices, face_vertex_normals)

        # normalize
        magnitudes = np.einsum("...i,...i", summed_normals, summed_normals) ** 0.5
        return summed_normals / magnitudes[:, None]

    # --- shading normals (face-varying, user-facing) ---

    def set_normals(
        self,
        hard_edge_angle: float      | None = None,
        hard_edges:      np.ndarray | None = None,
        angle_weighted:  bool              = True,
        area_weighted:   bool              = True,
    ) -> None:
        """Compute and store face-varying shading normals.

        These are independent from the geometric normals returned by
        ``get_vertex_normals()`` and ``get_face_normals()``.

        When ``normal_indices`` is ``None``, normals use vertex interpolation
        (indexed by ``self.indices``). When set, normals are face-varying
        and support hard edges.

        Args:
            hard_edge_angle: Angle threshold in degrees for auto-detecting
                hard edges.  If ``None`` and *hard_edges* is also ``None``,
                all edges are smooth.
            hard_edges: Explicit edge indices to treat as hard.
            angle_weighted: Weight by face-vertex corner angle (Maya default).
            area_weighted: Weight by face area (Maya default).
        """
        if hard_edges is None and hard_edge_angle is not None:
            hard_edges = _detect_hard_edges(self, hard_edge_angle)

        if hard_edges is None or hard_edges.size == 0:
            self.normals = self.get_vertex_normals(
                angle_weighted = angle_weighted,
                area_weighted  = area_weighted,
            )
            self.normal_indices = None
        else:
            self.normal_indices, num_normals = split_at_hard_edges(
                self.indices, self.counts, self.e2v, self.e2f, hard_edges
            )
            self.normals = _compute_shading_normals(
                self,
                self.normal_indices,
                num_normals,
                angle_weighted = angle_weighted,
                area_weighted  = area_weighted,
            )

    def recompute_normals(
        self,
        angle_weighted: bool = True,
        area_weighted:  bool = True,
    ) -> None:
        """Recompute shading normals from current positions.

        Preserves the existing face-varying topology (hard edge splitting).
        Only updates the normal vectors.  Call this after deforming
        ``self.points`` (e.g. via morph targets or skin weights).

        Raises ``RuntimeError`` if normals have not been set.
        """
        if self.normals is None:
            raise RuntimeError("No normals to recompute. Call set_normals() first.")

        if self.normal_indices is None:
            self.normals = self.get_vertex_normals(
                angle_weighted = angle_weighted,
                area_weighted  = area_weighted,
                force          = True,
            )
        else:
            self.normals = _compute_shading_normals(
                self,
                self.normal_indices,
                self.normals.shape[0],
                angle_weighted = angle_weighted,
                area_weighted  = area_weighted,
            )

    def get_tangent_space(
        self,
        uvdata: "UVData",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute MikkTSpace tangent vectors.

        Uses stored ``self.normals`` if available, otherwise falls back
        to smooth ``get_vertex_normals()``.

        The output is indexed by ``uvdata.indices`` (face-varying,
        matching UV topology).

        Args:
            uvdata: UV map providing texture coordinates.

        Returns:
            tangents:   (num_uv_verts, 4) xyz = tangent, w = bitangent sign.
            bitangents: (num_uv_verts, 3) reconstructed bitangents.
        """
        from cgmath.geometry.utils._numba._tangent_space import (
            _compute_mikktspace_triangles,
        )

        _require_no_ngons("get_tangent_space", self.counts)

        # resolve normals
        if self.normals is not None:
            shading_normals = np.asarray(self.normals)
            nrm_indices = (
                self.normal_indices if self.normal_indices is not None else self.indices
            )
        else:
            shading_normals = self.get_vertex_normals()
            nrm_indices     = self.indices

        # triangulate if needed
        tri_mesh        = self
        tri_uv          = uvdata
        tri_nrm_indices = nrm_indices

        has_quads = np.any(self.counts == 4)
        if has_quads:
            tri_mesh   = self.copy()
            tri_uv     = uvdata.copy()

            rules      = self.get_triangulate_rules(method="fast")
            expand_map = build_tri_expand_map(self.counts, rules.rules)

            tri_mesh.triangulate(rules=rules)
            tri_uv.triangulate(rules=rules)

            tri_nrm_indices = nrm_indices[expand_map]

        # Numba kernel: per-triangle tangent + bitangent
        face_T, face_B = _compute_mikktspace_triangles(
            tri_mesh.points,
            tri_mesh.geometry,
            tri_uv.points,
            tri_uv.geometry,
        )

        # angle-weighted accumulation to UV vertices
        face_vertex_angles = tri_mesh.get_face_vertex_angles()
        num_uv_verts       = tri_uv.point_count

        fv_T = np.repeat(face_T, tri_mesh.counts, axis=0)
        fv_B = np.repeat(face_B, tri_mesh.counts, axis=0)
        fv_T *= face_vertex_angles[:, None]
        fv_B *= face_vertex_angles[:, None]

        summed_T = np.zeros((num_uv_verts, 3))
        summed_B = np.zeros((num_uv_verts, 3))
        np.add.at(summed_T, tri_uv.indices, fv_T)
        np.add.at(summed_B, tri_uv.indices, fv_B)

        # resolve normals at UV vertices
        uv_to_nrm = build_uv_to_normal_map(
            tri_uv.indices,
            tri_nrm_indices,
            num_uv_verts,
        )
        N = shading_normals[uv_to_nrm]

        # Gram-Schmidt orthonormalization
        with np.errstate(divide="ignore", invalid="ignore"):
            dot_NT = np.einsum("ij,ij->i", N, summed_T)
            T      = summed_T - N * dot_NT[:, None]
            mag_T  = np.einsum("ij,ij->i", T, T) ** 0.5
            T /= mag_T[:, None]

            # Independently project accumulated B onto tangent plane
            dot_NB = np.einsum("ij,ij->i", N, summed_B)
            B      = summed_B - N * dot_NB[:, None]
            mag_B  = np.einsum("ij,ij->i", B, B) ** 0.5
            B /= mag_B[:, None]

            # Determine bitangent sign from UV triangle orientation
            # (matching MikkTSpace reference), not from accumulated B.
            geom      = tri_uv.geometry
            ds1       = tri_uv.points[geom[:, 1], 0] - tri_uv.points[geom[:, 0], 0]
            dt1       = tri_uv.points[geom[:, 1], 1] - tri_uv.points[geom[:, 0], 1]
            ds2       = tri_uv.points[geom[:, 2], 0] - tri_uv.points[geom[:, 0], 0]
            dt2       = tri_uv.points[geom[:, 2], 1] - tri_uv.points[geom[:, 0], 1]
            face_det  = ds1 * dt2 - ds2 * dt1
            face_sign = np.where(face_det > 0, 1.0, -1.0)

            # Propagate per-face sign to UV vertices via majority vote.
            fv_sign     = np.repeat(face_sign, tri_mesh.counts)
            uv_sign_sum = np.zeros(num_uv_verts)
            np.add.at(uv_sign_sum, tri_uv.indices, fv_sign)
            sign            = np.sign(uv_sign_sum)
            sign[sign == 0] = -1.0

            bitangents = B

        # handle degenerates -- preserve UV-determinant sign
        degenerate_T                            = ~np.isfinite(T).all(axis=1)
        degenerate_B                            = ~np.isfinite(bitangents).all(axis=1)
        T[degenerate_T]                         = [1, 0, 0]
        bitangents[degenerate_T | degenerate_B] = [0, 1, 0]

        tangents = np.column_stack([T, sign])
        return tangents, bitangents

    def get_edge_lengths(self) -> np.ndarray:
        """Returns the edge lengths."""
        if self._edge_lengths is None:
            points             = self.points
            edge_pairs         = self.e2v
            point_a_idxs       = edge_pairs[:, 0]
            point_b_idxs       = edge_pairs[:, 1]
            points_a           = points[point_a_idxs]
            points_b           = points[point_b_idxs]
            edges              = points_b - points_a
            self._edge_lengths = np.einsum("...i,...i", edges, edges) ** 0.5
        return self._edge_lengths

    def get_edge_normals(self) -> np.ndarray:
        if self._edge_normals is None:
            edge_face_pairs = self.e2f
            edges           = self.edge_count

            self._edge_normals = np.ndarray((edges, 3))

            areas   = self.get_face_areas()
            normals = self.get_face_normals()

            face_a  = edge_face_pairs[:, 0]
            face_b  = edge_face_pairs[:, 1]

            area_a                 = areas[face_a]
            area_b                 = areas[face_b]
            area_a[face_a == -1]   = 0  # set to 0 if no face is assigned
            area_b[face_b == -1]   = 0  # set to 0 if no face is assigned
            area_sum               = area_a + area_b

            weight_a               = area_a / area_sum
            weight_b               = area_b / area_sum
            normal_a               = normals[face_a]
            normal_b               = normals[face_b]
            normal_a[face_a == -1] = 0  # set to [0,0,0] if no face is assigned
            normal_b[face_b == -1] = 0  # set to [0,0,0] if no face is assigned

            self._edge_normals = (normal_a * weight_a[:, None]) + (
                normal_b * weight_b[:, None]
            )
            self._edge_normals = (
                self._edge_normals / np.linalg.norm(self._edge_normals, axis=1)[:, None]
            )

        return self._edge_normals

    # --- edit mesh

    def delete_faces(self, faces, exclude=False):
        """deletes faces at given face indices"""
        # find the reverse mapping and delete everyghing not in faces
        if not exclude:
            arange = np.arange(self.counts.size)
            faces  = np.setdiff1d(arange, faces)
        else:
            faces = np.unique(faces)
            faces = faces[(faces >= 0) & (faces < self.face_count)]

        f2v = self.f2v[faces]
        idx = np.unique(f2v)

        self.reset_cached_data()
        self.counts = self.counts[faces]
        self.points = self.points[idx[idx >= 0]]
        _, self.indices = np.unique(f2v[f2v >= 0], return_inverse=True)

        # reset components
        self.reset_cached_data()
        self.normals        = None
        self.normal_indices = None

    def __iadd__(self, other):
        """inplace add of two MeshData objects"""

        # combine
        if isinstance(other, MeshData):
            self.union(other)

        # morph target
        elif all(hasattr(other, x) for x in ["indices", "offsets"]):
            self.points[other.indices] += other.offsets

        # assume arithmetic offset of points
        else:
            self.points += other

        return self

    def __add__(self, other):
        """addition of two MeshData objects"""
        new = self.copy()
        return new.__iadd__(other)

    def __radd__(self, other):
        """
        right-hand side addition of two MeshData objects
        allows to use builtin sum()
        """
        if other == 0:
            return self
        return self.__add__(other)

    def __isub__(self, other):
        """inplace Streams of two MeshData objects"""

        # difference
        if isinstance(other, MeshData):
            self.difference(other)

        # morph target
        elif all(hasattr(other, x) for x in ["indices", "offsets"]):
            self.points[other.indices] -= other.offsets

        # assume arithmetic offset of points
        else:
            self.points -= other

        return self

    def __sub__(self, other):
        """subtraction of two MeshData objects"""
        new = self.copy()
        return new.__isub__(other)

    def __rsub__(self, other):
        """
        right-hand side subtraction of two MeshData objects
        """
        if other == 0:
            return self
        return self.__sub__(other)

    def __imul__(self, other):
        """inplace multiplication of two MeshData objects"""

        # assume arithmetic multiplication
        self.points *= other

        return self

    def __mul__(self, other):
        """multiplication of two MeshData objects"""
        new = self.copy()
        return new.__imul__(other)

    def __rmul__(self, other):
        """
        right-hand side multiplication of two MeshData objects
        """
        return self.__mul__(other)

    def union(self, *args, world_space=False):
        """combines faces of given Mesh objects to self (this is not a boolean)"""

        # bring self points to worldspace
        if world_space:
            points4 = np.dot(self.points4, self.matrix)
        else:
            points4 = self.points4

        # bring other objects points into this object's space
        for obj in args:
            self.indices = np.concatenate(
                [self.indices, obj.indices + points4.shape[0]]
            )
            self.counts = np.concatenate([self.counts, obj.counts])

            # bring obj points to worldspace
            if world_space:
                points_ = np.dot(obj.points4, obj.matrix)  # - self.matrix[3]
                points4 = np.concatenate([points4, points_])
            else:
                points4 = np.concatenate([points4, obj.points4])

        # bring back points to local space
        inv_matrix  = self.get_inverse_matrix()
        self.points = np.dot(points4, inv_matrix.T)[:, : self.points.shape[1]]

        # reset components
        self.reset_cached_data()
        self.normals        = None
        self.normal_indices = None

    def difference(self, *args, tolerance=1e-6, world_space=False):
        """removes overlapping faces of given Mesh objects from self (this is not a boolean)"""

        # build a source kdTree
        if world_space:
            source = self.copy()
            source.to_identity()
        else:
            source = self

        # find matching vertices for each obj to remove
        matched_indices = []
        for obj in args:
            if world_space:
                target = obj.copy()
                target.to_identity()
            else:
                target = obj

            tree = cKDTree(target.points)

            distances, idx = tree.query(source.points)
            indices = np.where(distances <= tolerance)[0]
            matched_indices.append(indices)

        # use the matched data to extract a negative of self
        vertices = np.unique(matched_indices)
        if vertices.size:
            source = source.from_vertices(vertices, contained=True, exclude=True)

            self.indices = source.indices
            self.counts  = source.counts
            self.points  = source.points

            # reset components
            self.reset_cached_data()
            self.normals        = None
            self.normal_indices = None

    def detach_faces(self, indices=None):
        """detaches faces"""

        if indices is None:
            # make each face its own island
            self.points  = self.points[self.indices]
            self.indices = np.arange(self.indices.size)

        else:
            # keep indices in range
            indices = np.unique(indices)
            indices = indices[(indices >= 0) & (indices < self.face_count)]

            # if we're given the whole thing, just do a full detach
            if indices[0] == 0 and indices[-1] == self.face_count - 1:
                return self.detach_faces(indices=None)

            # find the points and geometry on a fully detached mesh
            new_faces = self.copy()
            new_faces.detach_faces()

            geometry   = new_faces.geometry[indices]
            new_idx    = np.unique(geometry)
            new_idx    = new_idx[new_idx >= 0]
            new_points = new_faces.points[new_idx]

            # cleanup existing points by using the existing `from_faces` method
            new_self = self.from_faces(indices, exclude=True)

            # use the cleaned up mesh to establish the point offset
            point_offset = np.arange(new_idx.size) + new_self.points.shape[0]
            new_geometry = indices_replace(geometry, new_idx, point_offset)

            # combined the new points and geometry to self
            mask          = np.zeros(self.face_count, dtype=bool)
            mask[indices] = True

            self.geometry[~mask] = new_self.geometry
            self.geometry[mask]  = new_geometry

            self.points = np.concatenate([new_self.points, new_points])
            self.indices, self.counts = matrix_to_stream(self.geometry)

        # reset cached geometry
        self.reset_cached_data()
        self.normals        = None
        self.normal_indices = None

    def merge(self):
        """merges overlapping points"""

        # combine points and their indices
        indices, v0, v1 = point_row_overlaps(self.points, tolerance=1e-6)
        self.indices = indices_replace(self.indices, v0, v1, collapse=True)
        self.points  = self.points[indices]

        # remove overlapping faces
        matrix = stream_to_matrix(self.indices, self.counts)
        faces  = matrix_row_overlaps(matrix, exact=False)[0]
        self.indices, self.counts = matrix_to_stream(matrix[faces])

        # reset components
        self.reset_cached_data()
        self.normals        = None
        self.normal_indices = None

    def contains_vertices(self, vertices, contained=True, exclude=False):
        """returns the mesh's faces which contains the given vertex indices"""

        vertices = np.asarray(vertices)

        if contained:
            vertices = np.concatenate([vertices, [-1]])
            match = np.all(np.isin(self.f2v, vertices), axis=1)
        else:
            vertices = vertices[vertices >= 0]
            match = np.any(np.isin(self.f2v, vertices), axis=1)

        # reverse matched faces if excluding
        if exclude:
            return np.where(~match)[0]

        return np.where(match)[0]

    def get_subset_rules(self, vertices, exclude=False, min_count=3):
        """returns the local vertex slots each face keeps for a vertex subset

        Where contains_vertices is all or nothing per face, this keeps the
        selected vertices of a partially selected face and drops the rest, so
        a quad missing one vertex survives as a triangle.

        Args:
            vertices: Vertex indices to keep. Generated from self.geometry, so
                on a UVData these are uv point indices.
            exclude: Keep everything except the given vertices.
            min_count: Faces left holding fewer slots than this are dropped.

        Returns:
            An ``(num_faces, max(counts))`` array of the local vertex slots
            each face keeps, in source order, ``-1`` padded. A row of all
            ``-1`` drops the face. Slots are face local, so the same rules
            apply to a :class:`MeshData` and to its corresponding
            :class:`UVData` even though their indices differ across UV seams.
        """
        geometry = self.geometry
        vertices = np.unique(np.atleast_1d(np.asarray(vertices)))

        keep = np.isin(geometry, vertices)
        if exclude:
            keep = ~keep
        keep &= geometry >= 0

        # a face keeping fewer slots than min_count is dropped entirely
        keep[keep.sum(axis=1) < min_count] = False

        # a stable sort moves the kept slots to the front while holding their
        # source order, which is what preserves the winding
        rules = np.argsort(~keep, axis=1, kind="stable")
        slot  = np.arange(geometry.shape[1])
        return np.where(slot < keep.sum(axis=1)[:, None], rules, -1).astype(
            geometry.dtype
        )

    def rules_to_vertices(self, rules):
        """returns the vertices a set of subset rules keeps

        These are the source vertices from_vertices carries into the new
        mesh, ascending, so slot i of the result is vertex i of that mesh.
        """
        rules = np.asarray(rules)
        kept  = rules >= 0
        stream = np.take_along_axis(self.geometry, np.where(kept, rules, 0), axis=1)[
            kept
        ]
        return np.unique(stream)

    def from_vertices(self, vertices, contained=True, exclude=False):
        """rebuilds a new mesh from given vertex indices

        Args:
            vertices: A vertex index, a 1d sequence of them, or a 2d array of
                subset rules from get_subset_rules. The flat forms take whole
                faces, as they always have; rules keep part of a face, so a
                quad missing one vertex comes back as a triangle.
            contained: Faces must lie entirely inside the selection rather
                than merely touch it. Not accepted alongside rules.
            exclude: Rebuild from the faces the selection did not match. Not
                accepted alongside rules.
        """
        vertices = np.asarray(vertices)

        if vertices.ndim > 1:
            if not contained or exclude:
                raise ValueError(
                    "contained and exclude do not apply to subset rules, "
                    "which already describe exactly what is kept"
                )
            return self._from_rules(vertices)

        faces = self.contains_vertices(np.unique(vertices), contained=contained)
        return self.from_faces(faces, exclude=exclude)

    def from_faces(self, faces, exclude=False):
        """rebuilds a new mesh from given face indices"""
        new = self.copy()
        new.delete_faces(faces, exclude=(not exclude))
        return new

    def _from_rules(self, rules):
        """rebuilds a new mesh from get_subset_rules output"""
        rules  = np.asarray(rules)
        counts = self.counts
        width  = int(counts.max()) if counts.size else 0

        if rules.shape != (counts.size, width):
            raise ValueError(
                f"Subset rules must be shaped {(counts.size, width)} to match "
                f"this topology, got {tuple(rules.shape)}"
            )

        kept = rules >= 0

        # Face local slots are what let one set of rules drive a MeshData and
        # its UVData. This is the check that catches rules built against faces
        # of a different size -- without it the two results silently stop
        # being parallel instead of failing.
        if np.any(kept & (rules >= counts[:, None])):
            raise ValueError(
                "Subset rules select vertex slots this topology does not have"
            )

        # width is one past the largest legal slot, so it cannot collide
        ordered = np.sort(np.where(kept, rules, width), axis=1)
        if np.any((np.diff(ordered, axis=1) == 0) & (ordered[:, :-1] < width)):
            raise ValueError("Subset rules select the same vertex slot twice")

        stream = np.take_along_axis(self.geometry, np.where(kept, rules, 0), axis=1)[
            kept
        ]
        surviving, indices = np.unique(stream, return_inverse=True)

        new         = self.copy()
        new.counts  = kept.sum(axis=1)[kept.any(axis=1)].astype(counts.dtype)
        new.points  = self.points[surviving]
        new.indices = indices
        # as delete_faces: a per face vertex normal cannot outlive a change to
        # how many vertices the face has
        new.normals        = None
        new.normal_indices = None
        new.reset_cached_data()
        return new

    def sample(
        self,
        obj:                 Union["MeshData", np.ndarray],
        method:              SampleMethod | str            = SampleMethod.BILINEAR,
        iteration_count:     int                           = 100,
        iteration_tolerance: float                         = 1e-8,
    ):
        """a humble saddle surface sampler

        Args:
            obj: query points or MeshData whose vertices are used as query points.
            method: SampleMethod.BILINEAR (flat patches) or SampleMethod.BEZIER
                    (PN Quad bicubic Bezier patches using vertex normals for
                    higher accuracy on smooth meshes).
            iteration_count: max Newton iterations.
            iteration_tolerance: convergence tolerance.
        """
        _require_no_ngons("sample", self.counts)
        method = SampleMethod(method)

        if isinstance(obj, MeshData):
            points = np.copy(obj.points)
        else:
            points = np.asarray(obj)

        # gather normals if this is a 3D mesh (not a 2D UV map)
        normals = None
        if self.points.shape[1] > 2:
            normals = self.get_vertex_normals(angle_weighted=True, area_weighted=False)

        surface_normals = None
        if method == SampleMethod.BEZIER and normals is not None:
            surface_normals = normals

        return sample(
            points,
            self.points,
            self.geometry,
            normals             = normals,
            surface_normals     = surface_normals,
            iteration_count     = iteration_count,
            iteration_tolerance = iteration_tolerance,
        )

    # Skip BVH for tiny meshes - building + traversal overhead would cost
    # more than the brute-force linear scan saves.
    _BVH_MIN_FACES = 64

    @property
    def bvh_bilinear(self):
        """Lazily-built BVH over per-face vertex AABBs.

        Used to accelerate :meth:`raycast` with ``method="bilinear"``.
        Returns ``None`` for small meshes (<= 64 faces) where the BVH
        traversal overhead exceeds the brute-force scan cost.
        """
        if self._bvh_bilinear is not None:
            return self._bvh_bilinear
        if self.face_count <= self._BVH_MIN_FACES or self.points.shape[1] != 3:
            return None

        from cgmath.geometry.utils._numba._bilinear import _compute_face_aabbs
        from cgmath.geometry.utils._numba._bvh import build_bvh

        geom = np.asarray(self.geometry, dtype=np.int32)
        if geom.shape[1] < 4:
            pad                     = np.full((geom.shape[0], 4), -1, dtype=np.int32)
            pad[:, : geom.shape[1]] = geom
            geom                    = pad
        pts = np.ascontiguousarray(self.points, dtype=np.float64)
        aabb_min, aabb_max = _compute_face_aabbs(geom, pts)
        self._bvh_bilinear = build_bvh(aabb_min, aabb_max, leaf_size=8)
        return self._bvh_bilinear

    @property
    def bvh_bezier(self):
        """Lazily-built BVH over PN-Quad bicubic Bezier control-point AABBs.

        Used to accelerate :meth:`raycast` with ``method="bezier"``.
        Returns ``None`` for small meshes (<= 64 faces).  Cached
        independently from :attr:`bvh_bilinear` because the per-face
        bounds differ (Bezier uses control points, not vertices).
        """
        if self._bvh_bezier is not None:
            return self._bvh_bezier
        if self.face_count <= self._BVH_MIN_FACES or self.points.shape[1] != 3:
            return None

        from cgmath.geometry.utils._numba._bilinear import (
            _compute_all_pn_quad_cps,
            _compute_bezier_face_aabbs,
        )
        from cgmath.geometry.utils._numba._bvh import build_bvh

        geom = np.asarray(self.geometry, dtype=np.int32)
        if geom.shape[1] < 4:
            pad                     = np.full((geom.shape[0], 4), -1, dtype=np.int32)
            pad[:, : geom.shape[1]] = geom
            geom                    = pad
        pts = np.ascontiguousarray(self.points, dtype=np.float64)
        normals = np.ascontiguousarray(
            self.get_vertex_normals(angle_weighted=True, area_weighted=False),
            dtype=np.float64,
        )
        cps = _compute_all_pn_quad_cps(geom, pts, normals)
        aabb_min, aabb_max = _compute_bezier_face_aabbs(cps)
        self._bvh_bezier = build_bvh(aabb_min, aabb_max, leaf_size=8)
        return self._bvh_bezier

    def raycast(
        self,
        origins:      MeshData     | np.ndarray,
        directions:   np.ndarray   | None       = None,
        method:       SampleMethod | str        = SampleMethod.BILINEAR,
        forward_only: bool                      = True,
        twosided:     bool                      = True,
    ) -> "RaycastData":
        """ray-mesh intersection on bilinear or bicubic Bezier patches

        Args:
            origins: ray origins or MeshData whose vertices are used as origins.
            directions: ray directions. If None and origins is MeshData, uses
                        origin vertex normals.
            method: SampleMethod.BILINEAR (flat patches) or SampleMethod.BEZIER
                    (PN Quad bicubic Bezier patches using vertex normals for
                    higher accuracy on smooth meshes).
            forward_only: if False, also cast backward and keep the closer hit.
            twosided: when False, back-face hits are rejected.

        Implementation note: for meshes larger than 64 faces, a per-method
        BVH is lazily built (see :attr:`bvh_bilinear` / :attr:`bvh_bezier`)
        and reused across calls, giving 10-30x speedup vs the brute-force
        per-face scan.  The BVH is invalidated automatically by
        :meth:`reset_cached_data` (called from :meth:`triangulate`,
        :meth:`quadrangulate`, :meth:`subdivide`, etc.).
        """
        from cgmath.geometry._saddle_surface import raycast as _raycast

        _require_no_ngons("raycast", self.counts)
        method = SampleMethod(method)

        # gather normals if this is a 3D mesh
        if self.points.shape[1] > 2:
            normals = self.get_vertex_normals(angle_weighted=True, area_weighted=False)
        else:
            normals = None

        surface_normals = None
        if method == SampleMethod.BEZIER and normals is not None:
            surface_normals = normals

        # if origin is a MeshData obj, use the points as origin
        if isinstance(origins, MeshData):
            o = np.copy(origins.points)
        else:
            o = np.asarray(origins)

        # if directions is None and origin is MeshData, use origin vertex normals
        if directions is None and isinstance(origins, MeshData):
            d = origins.get_vertex_normals()

        # if directions is MeshData, us direction vertex normals
        elif isinstance(directions, MeshData):
            d = directions.get_vertex_normals()

        # if directions is a single vector, tile it to match origin
        else:
            d = np.asarray(directions)
            if d.ndim == 1:
                d = np.tile(d, (o.shape[0], 1))

        # Pick the matching cached BVH (or None - falls back to brute force).
        bvh = self.bvh_bezier if surface_normals is not None else self.bvh_bilinear

        return _raycast(
            self.points,
            self.points,
            self.geometry,
            o,
            d,
            normals         = normals,
            surface_normals = surface_normals,
            forward_only    = forward_only,
            twosided        = twosided,
            bvh             = bvh,
        )

    def to_identity(self) -> None:
        """sets the matrix to identity and transforms the points accordingly"""
        self.points = np.dot(self.points4, self.matrix)[:, : self.points.shape[1]]
        self.matrix = np.eye(4)

    def get_inverse_matrix(self) -> np.ndarray:
        """computes the inverse matrix"""
        scale_components   = np.sum(self.matrix[:3, :3] ** 2, axis=1)
        inv_matrix         = np.eye(4)
        inv_matrix[:3, :3] = self.matrix[:3, :3] / scale_components[:, None]
        inv_matrix[3, :3]  = -np.dot(inv_matrix[:3, :3], self.matrix[3, :3])
        return inv_matrix

    def get_extent(self, per_face: bool = False) -> Tuple[np.ndarray]:
        """Returns 2 tuples that represents the extent of this mesh."""

        # return the global bbx
        if not per_face:
            return self.points.min(axis=0), self.points.max(axis=0)

        # return the bbx of each face
        indexed_points                       = self.points[self.geometry]
        indexed_points[indexed_points == -1] = np.nan

        min_coords = np.nanmin(indexed_points, axis=1)
        max_coords = np.nanmax(indexed_points, axis=1)
        return min_coords, max_coords

    def apply_morph_target(self, target_data) -> None:
        """Applies a morph target to this mesh.

        Args:
            target_data: A morph target data object.
        """
        self.points[target_data.indices] += target_data.offsets

    # --------------------------------- SYMMETRY --------------------------------- #

    def get_asymmetric_points(
        self,
        pivot:     float              = 0.0,
        axis:      int                = 0,
        tolerance: Union[None, float] = None,
    ):
        """returns the vertex indices that fail the symmetry test"""

        # set a default tolerance using the max deviation
        if tolerance is None:
            tolerance = self.get_symmetry_deviation(pivot=pivot, axis=axis)[0]

        d = self.get_symmetry_map(pivot=pivot, axis=axis)[0]
        return np.where(d > tolerance)[0]

    def get_center_points(
        self,
        pivot:     float              = 0.0,
        axis:      int                = 0,
        tolerance: Union[None, float] = None,
    ) -> np.ndarray:
        """
        returns the indices of the center points
        """

        # set a default tolerance using the max deviation
        if tolerance is None:
            tolerance = self.get_symmetry_deviation(pivot=pivot, axis=axis)[-1]

        indices = np.abs(self.points[:, axis] - pivot) <= tolerance
        return np.where(indices)[0]

    @staticmethod
    def _get_symmetry_map(points: np.ndarray, pivot: float = 0.0, axis: int = 0):
        """returns the symmetric mapping for each vertex"""
        source_points = points.copy()
        source_points[:, axis] -= pivot
        mirror_points = source_points.copy()
        mirror_points[:, axis] *= -1

        # find closest point against mirror image
        tree = cKDTree(source_points)
        return tree.query(mirror_points)

    def get_symmetry_map(
        self, pivot: float = 0.0, axis: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """returns the symmetric mapping for each vertex"""
        return self._get_symmetry_map(self.points, pivot=pivot, axis=axis)

    def get_symmetry_deviation(self, pivot: float = 0.0, axis: int = 0) -> np.ndarray:
        """returns the min/max distance in the symmetry mapping"""
        d = self.get_symmetry_map(pivot=pivot, axis=axis)[0]
        return np.array([d.min(), d.max()])

    def fix_symmetry(
        self,
        pivot:     float              = 0.0,
        axis:      int                = 0,
        side:      float              = 1.0,
        tolerance: Union[None, float] = None,
        force:     bool               = False,
    ) -> bool:
        """fixes the mesh point symmetry, return True/False if the result is symmetrical"""

        # TODO: Make a separate MeshSymmetryData object to store symmetry and then use it to mirror points.

        # --- BOTH SIDES SPLIT THE DIFFERENCE SYMMETRY --- #
        # if side == 0, we try split the difference and fix symmetry on both sides
        if side == 0:
            positive_side = self.copy()
            negative_side = self.copy()
            test0 = positive_side.fix_symmetry(
                pivot=pivot, axis=axis, tolerance=tolerance, force=force, side=-1
            )
            test1 = negative_side.fix_symmetry(
                pivot=pivot, axis=axis, tolerance=tolerance, force=force, side=1
            )

            if force or (test0 and test1):
                self.points = (positive_side.points + negative_side.points) * 0.5

                return True
            return False

        # --- SIDE SPECIFIED SYMMETRY --- #

        # set a default tolerance using the max deviation
        if tolerance is None:
            tolerance = self.get_symmetry_deviation()[-1]

        # get points
        duplicate = self.copy()

        source_points = duplicate.points.copy()
        source_points[:, axis] -= pivot
        mirror_points = source_points.copy()
        mirror_points[:, axis] *= -1

        # find closest point against mirror image
        tree = cKDTree(source_points)
        _, indices = tree.query(mirror_points)

        # fix centerline
        center                      = np.where(np.abs(source_points[:, axis]) <= tolerance)
        source_points[center, axis] = 0.0

        # set positive side to match negative
        if side >= 0.0:
            match_indices = np.where(source_points[:, axis] < -tolerance)

        # set negative side to match negative
        else:
            match_indices = np.where(source_points[:, axis] > tolerance)

        source_points[match_indices] = mirror_points[indices[match_indices]]

        # do a final symmetry test with zero tolerance for error
        duplicate.points = source_points
        if force or duplicate.symmetrical(
            pivot     = pivot,
            axis      = axis,
            tolerance = EPSILON,
        ):
            source_points[:, axis] += pivot
            self.points = source_points
            return True

        return False

    def symmetrical(
        self,
        pivot:         float = 0.0,
        axis:          int   = 0,
        tolerance:     float = 1e-6,
        check_topology       = True,
        check_normals        = True,
    ) -> bool:
        # get symmetry mapping
        distances, mi = self.get_symmetry_map(pivot=pivot, axis=axis)

        # are all indices used?
        if np.unique(mi).size != self.point_count:
            return False

        # are all distances are within tolerance?
        if not np.all(distances <= tolerance):
            return False

        # --- topology check --- #

        if not check_topology:
            return True

        # make sure mirrored vertices are connected to the same face indices
        source_neighbors = self.get_face_vertex_neighbors()
        source_neighbors = np.sort(source_neighbors, axis=1)
        mirror_neighbors = indices_replace(
            source_neighbors[mi], np.arange(self.point_count), mi
        )
        mirror_neighbors = np.sort(mirror_neighbors, axis=1)
        if not np.allclose(source_neighbors, mirror_neighbors):
            return False

        # test that the normals are pointing the right way
        if not check_normals:
            return True

        _require_no_ngons("symmetrical", self.counts)
        uv = np.ones((self.face_count, 2)) * 0.5
        U, V = bilinear_vectors(self.points, self.geometry, uv)

        # sum all the face normals
        source_normals                 = np.cross(U, V)  # compute normal
        source_normals                 = source_normals[self.v2f]
        source_normals[self.v2f == -1] = 0
        source_normals                 = np.sum(source_normals, axis=1)

        # gather the mirror normals
        mirror_normals = source_normals[mi]
        mirror_normals[:, axis] *= -1

        # will return True if this mesh is reasonably symmetrical
        delta = source_normals - mirror_normals
        return np.all(np.einsum("...i,...i", delta, delta) <= tolerance)

    # ------------------------------ OBJ interface ------------------------------- #
    @classmethod
    def load_obj(cls, filename: str) -> "MeshData":
        """a basic obj file reader to MeshData"""

        filename    = os.path.expanduser(filename)
        points      = []
        normals     = []
        vert_ids    = []
        normal_ids  = []
        vert_counts = []
        name        = None

        with open(filename, "r") as f:
            for line in f.readlines():
                if line.startswith("g "):
                    name = line.split(" ")[1]
                    if name == "default":
                        name = None

                elif line.startswith("vn "):
                    normals.append([float(x) for x in line.split()[1:]])

                elif line.startswith("v "):
                    points.append([float(x) for x in line.split(" ")[1:]])

                elif line.startswith("f "):
                    tokens       = line.split()[1:]
                    face_verts   = []
                    face_normals = []
                    for tok in tokens:
                        parts = tok.split("/")
                        face_verts.append(int(parts[0]))
                        if len(parts) >= 3 and parts[2]:
                            face_normals.append(int(parts[2]))
                    vert_ids.extend(face_verts)
                    normal_ids.extend(face_normals)
                    vert_counts.append(len(face_verts))

        points      = np.array(points)
        vert_ids    = np.array(vert_ids) - 1
        vert_counts = np.array(vert_counts)

        # re-insert the term "Shape" back into the name
        if name is not None:
            match = re.search(r"\d+$", name)
            if match:
                index = match.start()
                name  = f"{name[:index]}Shape{name[index:]}"
            else:
                name = f"{name}Shape"

        mesh = cls(points=points, indices=vert_ids, counts=vert_counts, name=name)

        if normals:
            mesh.normals = np.array(normals)
            if normal_ids:
                mesh.normal_indices = np.array(normal_ids) - 1

        return mesh

    def save_obj(self, filename: str) -> None:
        """writes a valid obj file from MeshData"""
        filename = os.path.expanduser(filename)

        name = self.name
        if name is not None:
            if bool(re.search("Shape[0-9]*$", self.name)):
                name = "".join(name.rpartition("Shape")[::2])

        with open(filename, "w") as f:
            f.write("# OBJ file\n")
            # f.write("g default\n")
            if name is not None:
                f.write(f"g {name}\n")
            f.write("s 1\n")

            for p in self.points:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")

            for i, c in enumerate(self.counts):
                vals = [str(x) for x in (self.geometry[i, :c] + 1).tolist()]
                vals = " ".join(vals)
                f.write(f"f {vals}\n")

            f.write("\n")

    # ----------------------------------- GLB ------------------------------------ #

    # TODO scale_factor is prolly not the best name, think of a better way to handle unit conversions
    @classmethod
    def load_glb(cls, filename: str, scale_factor: float = 100.0) -> "MeshData":
        """a basic glb file reader to MeshData"""

        if trimesh is None:
            raise ImportError("trimesh is not installed")

        filename  = os.path.expanduser(filename)
        scene     = trimesh.load(filename)

        mesh_list = []
        for name, data in scene.geometry.items():
            points  = np.array(data.vertices) * scale_factor
            indices = np.array(data.faces)
            counts  = np.ones(indices.shape[0], dtype=int) * 3

            new = cls(points=points, indices=indices.ravel(), counts=counts, name=name)
            mesh_list.append(new)

        return mesh_list

    # ----------------------------------- FBX ------------------------------------ #

    @classmethod
    def load_fbx(cls, filename: str, name: str | None = None) -> "MeshData":
        """returns a single MeshData from an fbx file (first mesh if name is None)"""
        data = load_fbx(filename)
        if not data:
            raise RuntimeError("No meshes found in FBX file")
        if name is None:
            return data[0][0]
        for mesh_data, _ in data:
            if mesh_data.name == name:
                return mesh_data
        available = [m.name for m, _ in data]
        raise ValueError(f"Mesh '{name}' not found; available: {available}")

    # ------------------------------ USD interface ------------------------------- #

    @classmethod
    def from_prim(cls, prim: Any) -> "MeshData":
        """Construct a data object from a mesh prim.

        Args:
            prim: A prim to read the data from.

        Returns:
            A MeshData object.
        """
        mesh_api    = pxr().UsdGeom.Mesh(prim)
        vert_counts = np.array(mesh_api.GetFaceVertexCountsAttr().Get())
        vert_ids    = np.array(mesh_api.GetFaceVertexIndicesAttr().Get())
        points      = np.array(mesh_api.GetPointsAttr().Get())
        name        = prim.GetName()

        # read shading normals
        normals        = None
        normal_indices = None
        normals_attr   = prim.GetAttribute("primvars:normals")
        if normals_attr and normals_attr.HasValue():
            normals  = np.array(normals_attr.Get())
            idx_attr = prim.GetAttribute("primvars:normals:indices")
            if idx_attr and idx_attr.HasValue():
                normal_indices = np.array(idx_attr.Get(), dtype=int)

        return cls(
            points         = points,
            indices        = vert_ids,
            counts         = vert_counts,
            name           = name,
            normals        = normals,
            normal_indices = normal_indices,
        )

    def to_prim(self, prim: Any) -> None:
        """Streams data into a prim."""
        if not prim.GetTypeName():
            prim.SetTypeName(MESH_PRIM_TYPE)

        min_bbx, max_bbx = self.get_extent()
        mesh_api = pxr().UsdGeom.Mesh(prim)
        mesh_api.CreateFaceVertexCountsAttr().Set(self.counts)
        mesh_api.CreateFaceVertexIndicesAttr().Set(self.indices)
        mesh_api.CreatePointsAttr().Set(self.points)
        mesh_api.CreateExtentAttr().Set((min_bbx.tolist(), max_bbx.tolist()))

        # write shading normals
        if self.normals is not None:
            Sdf = pxr().Sdf
            attr = prim.CreateAttribute(
                "primvars:normals",
                Sdf.ValueTypeNames.Normal3fArray,
                False,
                Sdf.VariabilityVarying,
            )
            attr.Set(self.normals)
            attr.SetMetadata("interpolation", "faceVarying")

            if self.normal_indices is not None:
                idx_attr = prim.CreateAttribute(
                    "primvars:normals:indices",
                    Sdf.ValueTypeNames.IntArray,
                    False,
                    Sdf.VariabilityVarying,
                )
                idx_attr.Set(self.normal_indices)

    # ------------------------------ TRIANGULATION

    def get_triangulate_rules(
        self,
        method:     TriangulateMethod | str = TriangulateMethod.FAST,
        invert:     bool                    = False,
        ngons_only: bool                    = False,
    ) -> TriangulateRules:
        """generates triangulation rules according to specified method

        Args:
            method: Quad triangulation method (ignored when ``ngons_only=True``).
            invert: Flip quad split rules (0<->1). Has no effect on n-gons.
            ngons_only: When True, only triangulate n-gon faces (count > 4).
                Quads and triangles are left untouched (all rules stay ``-1``).
        """

        method            = TriangulateMethod(method)
        triangulate_rules = np.ones(self.counts.size, dtype=int) * -1

        # separate existing quads and triangles
        quad_ids  = np.where(self.counts == 4)[0]
        ngon_tris = {}

        if quad_ids.size > 0 and not ngons_only:
            quads  = self.geometry[quad_ids]
            arange = np.arange(quads.shape[0])[:, None]

            # make sure points are 3d (in case of uv triangulation)
            if self.points.shape[1] == 3:
                points = self.points
            else:
                points                            = np.zeros((self.points.shape[0], 3))
                points[:, : self.points.shape[1]] = self.points

            if method == TriangulateMethod.FAST:
                vtx = points[quads]

                # Squared diagonal lengths
                d02       = vtx[:, 0] - vtx[:, 2]
                d13       = vtx[:, 1] - vtx[:, 3]
                length_02 = np.einsum("...i,...i", d02, d02)
                length_13 = np.einsum("...i,...i", d13, d13)

                # Fudge factor: makes split 0-2 slightly preferred for exact ties
                fudge     = 0.999
                prefer_02 = length_02 < fudge * length_13

                # --- Split 0-2: triangles (1,2,0) and (0,2,3) ---
                n02_a        = np.cross(vtx[:, 2] - vtx[:, 1], vtx[:, 0] - vtx[:, 1])
                n02_b        = np.cross(vtx[:, 2] - vtx[:, 0], vtx[:, 3] - vtx[:, 0])
                agree_02     = np.einsum("ij,ij->i", n02_a, n02_b) > 0
                area_02_a    = np.einsum("...i,...i", n02_a, n02_a)
                area_02_b    = np.einsum("...i,...i", n02_b, n02_b)
                balance_02   = (area_02_a < 7 * area_02_b) & (area_02_b < 7 * area_02_a)
                can_split_02 = agree_02 & balance_02

                # --- Split 1-3: triangles (0,1,3) and (3,1,2) ---
                n13_a        = np.cross(vtx[:, 1] - vtx[:, 0], vtx[:, 3] - vtx[:, 0])
                n13_b        = np.cross(vtx[:, 1] - vtx[:, 3], vtx[:, 2] - vtx[:, 3])
                agree_13     = np.einsum("ij,ij->i", n13_a, n13_b) > 0
                area_13_a    = np.einsum("...i,...i", n13_a, n13_a)
                area_13_b    = np.einsum("...i,...i", n13_b, n13_b)
                balance_13   = (area_13_a < 7 * area_13_b) & (area_13_b < 7 * area_13_a)
                can_split_13 = agree_13 & balance_13

                # Decision: try preferred diagonal first, fall back to other, else -1
                rules                            = np.full(quads.shape[0], -1, dtype=int)
                rules[prefer_02 & can_split_02]  = 0
                rules[~prefer_02 & can_split_13] = 1

                # Fallback: if preferred failed, try the other diagonal
                fallback_02        = prefer_02 & ~can_split_02 & can_split_13
                fallback_13        = ~prefer_02 & ~can_split_13 & can_split_02
                rules[fallback_02] = 1
                rules[fallback_13] = 0

            elif method == TriangulateMethod.TRINITY:

                def tri_area(tri):
                    v0 = tri[:, 0] - tri[:, 1]
                    v1 = tri[:, 2] - tri[:, 1]
                    n  = np.cross(v0, v1)
                    return np.einsum("...i,...i", n, n)

                def double_minimum(t0, t1):
                    a0 = tri_area(t0)
                    a1 = tri_area(t1)
                    return np.minimum(a0, a1)

                # compute vertex and face centroid symmetry maps
                bc = points[quads].mean(axis=1)
                _, sym_fid = self._get_symmetry_map(bc)
                _, sym_vid = self.get_symmetry_map()

                # compute face minimum area under rule 0 (even, split vertex 0-2)
                t0    = points[quads[:, [0, 1, 2]]]
                t1    = points[quads[:, [0, 2, 3]]]
                area0 = double_minimum(t0, t1)

                t0    = points[quads[:, [1, 2, 3]]]
                t1    = points[quads[:, [1, 3, 0]]]
                area1 = double_minimum(t0, t1)

                # compute general splitting rules
                edge_indices = np.array([[0, 2], [1, 3]])
                rules        = np.where(area0 <= area1, 1, 0)
                edge_rules   = quads[arange, edge_indices[rules]]

                # check faces visited second
                verify = np.where(arange.ravel() > sym_fid)[0]

                # flip the rule when edges differ
                test0 = sym_vid[edge_rules[verify].ravel()]
                test0 = np.sort(test0.reshape(verify.size, 2), axis=1)
                test1 = np.sort(edge_rules[sym_fid[verify]], axis=1)
                flip  = ~np.all(test0 == test1, axis=1)

                rules[verify[flip]] = (rules[verify[flip]] + 1) % 2

            elif method == TriangulateMethod.EVEN:
                rules = np.zeros(quad_ids.size, dtype=int)

            elif method == TriangulateMethod.ODD:
                rules = np.ones(quad_ids.size, dtype=int)

            else:
                raise NotImplementedError(
                    "Must specify a supported triangulation method."
                )

            # invert the rules if requested (only affects quads)
            if invert:
                rules = np.where(rules == 0, 1, 0)

            triangulate_rules[quad_ids] = rules

        # --- N-gon CDT processing ---
        ngon_ids = np.where(self.counts > 4)[0]
        if ngon_ids.size > 0:
            from cgmath.geometry._cdt import triangulate_ngon

            # Build per-face hole mapping if hole metadata is available
            face_holes = {}
            if (
                self.hole_faces is not None
                and self.hole_counts is not None
                and self.hole_indices is not None
            ):
                hole_offset = 0
                for i, face_id in enumerate(self.hole_faces):
                    count = self.hole_counts[i]
                    hole_verts = list(
                        self.hole_indices[hole_offset : hole_offset + count]
                    )
                    hole_offset += count
                    face_holes.setdefault(face_id, []).append(hole_verts)

            pts = self.points
            for face_id in ngon_ids:
                face_verts = self.geometry[face_id]
                face_verts = face_verts[face_verts >= 0]

                holes = face_holes.get(int(face_id), None)
                if holes:
                    hole_set = set()
                    for h in holes:
                        hole_set.update(h)
                    outer      = [v for v in face_verts if v not in hole_set]
                    local_tris = triangulate_ngon(pts, outer, holes)
                else:
                    local_tris = triangulate_ngon(pts, face_verts)

                ngon_tris[int(face_id)] = local_tris

        return TriangulateRules(rules=triangulate_rules, ngon_tris=ngon_tris)

    def triangulate(
        self,
        rules:          Union[np.ndarray, TriangulateRules],
        preserve_order: bool                                = True,
    ) -> None:
        """
        triangulates the mesh according to the given method.
        if no method given, will used TriangulateMethod.FAST
        RULES : np.array([...], dtype=int) 0 = use even division (edge[0,2]),
                                           1 = use odd division (edge[1,3])

        Args:
            rules: Per-face triangulation rules.  Accepts a plain ndarray
                (backward-compatible, quads only) or a :class:`TriangulateRules`
                instance (quads + n-gons).
            preserve_order: If True (default), keep triangulated faces in
                the same relative order as the original faces (each quad is
                replaced in-place by its two triangles).  When False, groups
                all original triangles first, then all quad-derived triangles,
                which is faster but changes face ordering.
        """

        # Unpack TriangulateRules or use plain ndarray
        if isinstance(rules, TriangulateRules):
            ngon_tris = rules.ngon_tris
            rules     = rules.rules
        else:
            ngon_tris = {}

        # separate existing quads, triangles, and ngons
        tri_ids  = np.where(self.counts == 3)[0]
        quad_ids = np.where(self.counts == 4)[0]

        # filter quads to only those with an actual split rule (0 or 1)
        if quad_ids.size > 0:
            quad_rules_all = rules[quad_ids]
            active_mask    = quad_rules_all >= 0
            quad_ids       = quad_ids[active_mask]

        if quad_ids.size == 0 and not ngon_tris:
            return None

        # --- Quad triangulation ---
        if quad_ids.size > 0:
            quad_rules = rules[quad_ids]
            quads      = self.geometry[quad_ids]
            arange     = np.arange(quads.shape[0])[:, None]

            # Maya winding order:
            #   rule=0 (split 0-2): tri1 = (1,2,0), tri2 = (0,2,3)
            #   rule=1 (split 1-3): tri1 = (0,1,3), tri2 = (3,1,2)
            tri_rules = np.array([[1, 2, 0], [0, 1, 3]])
            triangleA = tri_rules[quad_rules]
            triangleA = quads[arange, triangleA]

            tri_rules = np.array([[0, 2, 3], [3, 1, 2]])
            triangleB = tri_rules[quad_rules]
            triangleB = quads[arange, triangleB]

            quad_tri_faces          = np.empty((2 * quad_rules.size, 3), dtype=int)
            quad_tri_faces[0::2, :] = triangleA
            quad_tri_faces[1::2, :] = triangleB
        else:
            triangleA      = np.empty((0, 3), dtype=int)
            triangleB      = np.empty((0, 3), dtype=int)
            quad_tri_faces = np.empty((0, 3), dtype=int)

        tris = (
            self.geometry[tri_ids, :3]
            if tri_ids.size > 0
            else np.empty((0, 3), dtype=int)
        )

        # --- N-gon triangulation: map local indices to global ---
        ngon_tri_faces_list = []
        for face_id, local_tris in ngon_tris.items():
            if len(local_tris) == 0:
                continue
            face_verts  = self.geometry[face_id]
            face_verts  = face_verts[face_verts >= 0]
            local_tris  = np.asarray(local_tris).reshape(-1, 3)
            global_tris = face_verts[local_tris]
            ngon_tri_faces_list.append(global_tris)

        if ngon_tri_faces_list:
            ngon_tri_faces = np.concatenate(ngon_tri_faces_list, axis=0)
        else:
            ngon_tri_faces = np.empty((0, 3), dtype=int)

        # recombine: use stream format to handle mixed face sizes
        # (skipped quads remain 4-vert, everything else is 3-vert)
        active_quad_set = set(quad_ids.tolist()) if quad_ids.size > 0 else set()
        ngon_set        = set(ngon_tris.keys())

        # precompute quad triangulations for fast lookup
        quad_tri_map = {}
        if quad_ids.size > 0:
            for i, qi in enumerate(quad_ids):
                quad_tri_map[qi] = (triangleA[i], triangleB[i])

        if preserve_order:
            new_indices = []
            new_counts  = []
            geom        = self.geometry
            for fi in range(self.counts.size):
                if fi in quad_tri_map:
                    triA, triB = quad_tri_map[fi]
                    new_counts.append(3)
                    new_indices.extend(triA.tolist())
                    new_counts.append(3)
                    new_indices.extend(triB.tolist())
                elif fi in ngon_set and len(ngon_tris[fi]) > 0:
                    face_verts = geom[fi]
                    face_verts = face_verts[face_verts >= 0]
                    for tri in np.asarray(ngon_tris[fi]).reshape(-1, 3):
                        new_counts.append(3)
                        new_indices.extend(face_verts[tri].tolist())
                else:
                    face_verts = geom[fi]
                    face_verts = face_verts[face_verts >= 0]
                    new_counts.append(len(face_verts))
                    new_indices.extend(face_verts.tolist())
        else:
            new_indices = []
            new_counts  = []
            geom        = self.geometry
            untouched = sorted(
                set(range(self.counts.size)) - active_quad_set - ngon_set
            )
            for fi in untouched:
                face_verts = geom[fi]
                face_verts = face_verts[face_verts >= 0]
                new_counts.append(len(face_verts))
                new_indices.extend(face_verts.tolist())
            for tri in quad_tri_faces:
                new_counts.append(3)
                new_indices.extend(tri.tolist())
            for tri in ngon_tri_faces:
                new_counts.append(3)
                new_indices.extend(tri.tolist())

        self.indices = np.array(new_indices, dtype=int)
        self.counts  = np.array(new_counts, dtype=int)

        # clear hole metadata -- mesh is now all triangles
        self.hole_faces   = None
        self.hole_counts  = None
        self.hole_indices = None

        # reset cached data
        self.reset_cached_data()
        self.normals        = None
        self.normal_indices = None

    # ------------------------------ QUADRANGULATION

    def get_quadrangulate_rules(
        self,
        planarity_weight:  float = 1.0,
        squareness_weight: float = 0.5,
        diagonal_weight:   float = 0.5,
        min_score:         float = 0.0,
        require_convex:    bool  = True,
    ) -> np.ndarray:
        """Identifies pairs of triangles to merge back into quads.  This is
        the inverse of :meth:`triangulate`.

        Solves the problem as a maximum-weight matching on the triangle dual
        graph: nodes = triangles, candidate edges = interior unique edges
        between two triangles, weight = quality of the resulting quad.  A
        greedy matching (O(E log E)) is used.

        Quad quality combines four geometric tests:

        * planarity            : dihedral across the candidate edge
        * squareness           : how close the four corners are to 90 deg
        * diagonal-dominance   : the removed edge should be longer than the
          sides (a true quad diagonal is longer than its sides)
        * convexity            : the merged quadrilateral must be convex
          when projected onto its average plane (hard reject)

        Args:
            planarity_weight:  Weight for the dihedral term.
            squareness_weight: Weight for the corner-angle term.
            diagonal_weight:   Weight for the diagonal-dominance term.
            min_score:         Discard candidates scoring below this.
            require_convex:    Reject merges that produce a concave quad.

        Returns:
            An ``(N, 4)`` int array; each row is
            ``(fa, fb, oa_slot, ob_slot)``:

            * ``fa``, ``fb``     : face indices of the two triangles to merge
            * ``oa_slot``        : local vertex slot (0, 1 or 2) of the
              "opposite" vertex in triangle ``fa`` (the vertex NOT on the
              shared edge).
            * ``ob_slot``        : same for triangle ``fb``.

            Slots are local to each face's vertex list, so the same rules
            can be applied to a :class:`MeshData` and to a corresponding
            :class:`UVData` with the same face topology.  May be empty.
        """

        score, fa, fb, cand = _score_quad_candidates(
            self,
            planarity_weight  = planarity_weight,
            squareness_weight = squareness_weight,
            diagonal_weight   = diagonal_weight,
            min_score         = min_score,
            require_convex    = require_convex,
        )
        if cand.size == 0:
            return np.empty((0, 4), dtype=int)

        pick   = quad_match_greedy(score, fa, fb, self.counts.size)
        chosen = cand[pick]
        if chosen.size == 0:
            return np.empty((0, 4), dtype=int)

        fa = self.ue2f[chosen, 0]
        fb = self.ue2f[chosen, 1]
        va = self.ue2v[chosen, 0]
        vb = self.ue2v[chosen, 1]

        tri_a = self.f2v[fa]
        tri_b = self.f2v[fb]
        # Opposite-vertex slot = the (single) True position in each row's
        # mask.  ``argmax`` returns the first index of the max value, which
        # is the True position when only one True exists.
        mask_a  = (tri_a != va[:, None]) & (tri_a != vb[:, None]) & (tri_a >= 0)
        mask_b  = (tri_b != va[:, None]) & (tri_b != vb[:, None]) & (tri_b >= 0)
        oa_slot = np.argmax(mask_a, axis=1)
        ob_slot = np.argmax(mask_b, axis=1)

        return np.column_stack([fa, fb, oa_slot, ob_slot]).astype(int)

    def quadrangulate(self, rules: np.ndarray | None = None) -> None:
        """Merges pairs of triangles into quads, in-place.

        Effectively the inverse of :meth:`triangulate`.  Triangles that
        cannot be paired are left untouched, as are any pre-existing quads
        or n-gons.  N-gon hole metadata (``hole_faces``) is remapped to the
        new face indices so that holes still reference their owners after
        the merge.

        Args:
            rules: ``(N, 4)`` int array of ``(fa, fb, oa_slot, ob_slot)``
                tuples produced by :meth:`get_quadrangulate_rules`.  When
                ``None`` (default), :meth:`get_quadrangulate_rules` is
                called with default arguments.

        Note:
            The rule format is purely face-local, so the same ``rules``
            array can be applied to a :class:`MeshData` and to its
            corresponding :class:`UVData` (which share face topology but
            may have different vertex indices due to UV seams).
        """

        if rules is None:
            rules = self.get_quadrangulate_rules()

        rules = np.asarray(rules, dtype=int).reshape(-1, 4)
        if rules.size == 0:
            return None

        # partner[face]      -> the face it merges with, or -1 if unmerged.
        # rule_for_face[fa]  -> rules-array row index where this face is fa.
        n_faces       = self.counts.size
        partner       = np.full(n_faces, -1, dtype=int)
        rule_for_face = np.full(n_faces, -1, dtype=int)
        for i in range(rules.shape[0]):
            fa_i                = int(rules[i, 0])
            fb_i                = int(rules[i, 1])
            partner[fa_i]       = fb_i
            partner[fb_i]       = fa_i
            rule_for_face[fa_i] = i

        new_indices: List[int] = []
        new_counts:  List[int] = []
        consumed = np.zeros(n_faces, dtype=bool)
        # Track where each old face ends up in the output - needed to
        # remap n-gon ``hole_faces`` after the merge shifts face indices.
        # Both triangles of a merged pair map to the new quad's index.
        new_face_id = np.full(n_faces, -1, dtype=int)
        out_idx     = 0

        for f in range(n_faces):
            if consumed[f]:
                continue
            p = partner[f]
            if p == -1:
                # unmerged face: copy as-is
                verts = self.f2v[f]
                verts = verts[verts >= 0]
                new_indices.extend(int(v) for v in verts)
                new_counts.append(int(self.counts[f]))
                new_face_id[f] = out_idx
                out_idx += 1
                consumed[f] = True
                continue

            # Determine which face is the rule's "fa" (only one of {f, p}).
            if rule_for_face[f] >= 0:
                fa, fb, ri = f, p, rule_for_face[f]
            else:
                fa, fb, ri = p, f, rule_for_face[p]

            oa_slot = int(rules[ri, 2])
            ob_slot = int(rules[ri, 3])

            tri_a   = self.f2v[fa]
            tri_a   = tri_a[tri_a >= 0]
            tri_b   = self.f2v[fb]
            tri_b   = tri_b[tri_b >= 0]

            oa = int(tri_a[oa_slot])
            va = int(tri_a[(oa_slot + 1) % 3])
            vb = int(tri_a[(oa_slot + 2) % 3])
            ob = int(tri_b[ob_slot])

            # ``[oa, va, ob, vb]`` is a cyclic rotation of the original
            # quad's winding (inherited from fa), so no normal-based flip
            # is needed - and the algorithm now works for 2D UV points too.
            new_indices.extend([oa, va, ob, vb])
            new_counts.append(4)
            new_face_id[fa] = out_idx
            new_face_id[fb] = out_idx
            out_idx += 1
            consumed[f] = True
            consumed[p] = True

        self.indices = np.array(new_indices, dtype=int)
        self.counts  = np.array(new_counts, dtype=int)

        # Remap n-gon ``hole_faces`` because merging shifts later face
        # indices.  N-gons are never selected as merge candidates, so every
        # entry must resolve to a non-negative new id.
        if self.hole_faces is not None and self.hole_faces.size > 0:
            remapped = new_face_id[self.hole_faces]
            if np.any(remapped < 0):
                raise RuntimeError(
                    "quadrangulate: hole_faces references a face that was "
                    "consumed by a merge - this should not happen because "
                    "n-gons are not eligible for merging."
                )
            self.hole_faces = remapped.astype(self.hole_faces.dtype)

        # reset cached data - topology has changed
        self.reset_cached_data()
        self.normals        = None
        self.normal_indices = None

    # ---------------------------------- SMOOTH ---------------------------------- #
    def smooth(
        self,
        neighbors:     np.ndarray | None       = None,
        indices:       np.ndarray | None       = None,
        iterations:    int        | np.ndarray = 1,
        receptions:    float      | np.ndarray = 0.5,
        contributions: float      | np.ndarray = 1.0,
    ):
        # get neighbors
        if neighbors is None:
            neighbors = self.get_edge_vertex_neighbors()

        # if indices None, smooth the whole mesh
        if indices is not None:
            if isinstance(iterations, int):
                iterations = np.ones(neighbors.shape[0], dtype=int) * iterations
            else:
                iterations = np.unique(iterations).astype(int)

            mask             = np.ones(neighbors.shape[0], dtype=bool)
            mask[indices]    = False
            iterations[mask] = 0

        # blur the points
        self.points = blur(
            self.points,
            neighbors,
            iterations    = iterations,
            receptions    = receptions,
            contributions = contributions,
        )

    # ------------------------ Catmull-Clark Subdivision ------------------------- #
    # https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface

    def subdivide(
        self,
        steps:        int   = 1,
        keep_borders: bool  = False,
        keep_edges:   bool  = False,
        fit:          bool  = False,
        tolerance:    float = 0.001,
        iterations:   int   = 10000,
        damping:      float = 0.5,
    ) -> None:
        _require_no_ngons("subdivide", self.counts)
        # use an iterative fit algorithm so the subd surface
        # passes through the source mesh coefficients (points)
        if fit and not keep_edges:
            points   = self.points.copy()  # original points
            max_peak = np.inf              # max peak set to infinity

            for _ in range(iterations):
                # copy self as a proxy and subdivide
                proxy = self.copy()
                proxy.subdivide(steps, keep_borders=keep_borders)

                # compute the delta between original points and subd
                delta = points - proxy.points[: points.shape[0]]
                mag   = np.einsum("...i,...i", delta, delta) ** 0.5
                peak  = mag.max()

                # stop if tolerance is met or no longer converging
                if peak < tolerance or peak >= max_peak:
                    break

                # set the new peak and apply displacement to points
                max_peak = peak
                self.points += delta * damping

            self.points  = proxy.points
            self.indices = proxy.indices
            self.counts  = proxy.counts
            self.reset_cached_data()
            self.normals        = None
            self.normal_indices = None

        # normal subdivision steps
        else:
            while steps > 0:
                steps -= 1
                new_points, new_indices, new_counts = subdivide_catmull_clark(
                    self.points,
                    self.indices,
                    self.counts,
                    self.f2v,
                    self.f2e,
                    self.e2v,
                    self.e2f,
                    self.v2f,
                    self.v2e,
                    keep_borders = keep_borders,
                    keep_edges   = keep_edges,
                )

                self.points  = new_points
                self.indices = new_indices
                self.counts  = new_counts
                self.reset_cached_data()

            self.normals        = None
            self.normal_indices = None

    def get_correspondence(self, mesh: "MeshData") -> None:
        """computes the point correspondence between MeshData objects"""

        if isinstance(mesh, MeshData):
            if mesh.counts.size == self.counts.size:
                indices = np.unique(self.indices)
                rows, cols = np.where(np.isin(self.geometry, indices))
                indices[self.geometry[rows, cols]] = mesh.geometry[rows, cols]

                return indices

            raise ValueError("MeshData objects must have the same topology.")

        raise TypeError(f"MeshData object expected. Got {type(mesh)}.")

    def to_uvdata(
        self,
        name:          str | None = None,
        axis:          str | Axis = Axis.Y,
        maintain_ratio            = False,
        swap_uvs                  = False,
    ) -> UVData:
        """creates an orthogonal projection of the mesh as a UVData object"""

        # if swap_uvs is True, swap the desired uv axes
        uv_axes = [0, 1]
        if swap_uvs:
            uv_axes = [1, 0]

        # project along an axis relevant
        points = np.zeros((self.point_count, 2))
        axis   = Axis(axis)
        if axis == Axis.X:
            points[:, uv_axes[0] - 1] = self.points[:, 1].copy()
            points[:, uv_axes[1] - 1] = self.points[:, 2].copy()
        elif axis == Axis.Y:
            points[:, uv_axes[0]] = self.points[:, 0].copy()
            points[:, uv_axes[1]] = self.points[:, 2].copy()
        elif axis == Axis.Z:
            points[:, uv_axes[0]] = self.points[:, 0].copy()
            points[:, uv_axes[1]] = self.points[:, 1].copy()
        else:
            raise ValueError("Invalid projection axis.")

        # create the UVData object
        uvdata = UVData(points=points, indices=self.indices, counts=self.counts)

        # normalize the UVData object, if requested conserve spatial ratio
        bbx_min, bbx_max = uvdata.get_extent()
        delta = bbx_max - bbx_min
        uvdata.normalize()

        if maintain_ratio:
            if delta[0] >= delta[1]:
                delta /= delta[0]
                offset = (1 - delta[1]) / 2
                uvdata.points[:, 1] *= delta[1]
                uvdata.points[:, 1] += offset
            else:
                delta /= delta[1]
                offset = (1 - delta[0]) / 2
                uvdata.points[:, 0] *= delta[0]
                uvdata.points[:, 0] += offset

        # give it a name if specified
        if name is not None:
            uvdata.name = name

        return uvdata


@dataclass(repr=False, eq=False)
class UVData(MeshData):
    """
    UV map topology treated as a Mesh object for resampling purposes
    """

    name: str = "map1"

    # --- cached attributes --- #
    _buffer     = None                    # render buffer
    _resolution = np.array([2048, 2048])  # render buffer resolution
    _renderer   = None
    _antialias  = False

    # buffer pixel counters
    _pixel_counts   = None
    _pixel_overlaps = None
    _pixel_ratios   = None

    def union(self, *args):
        """combines faces of given Mesh objects to self (this is not a boolean)"""
        for obj in args:
            self.indices = np.concatenate(
                [self.indices, obj.indices + self.point_count]
            )
            self.points = np.concatenate([self.points, obj.points])
            self.counts = np.concatenate([self.counts, obj.counts])

        # reset cached data
        self.reset_cached_data()

    def difference(self, *args, tolerance=1e-6):
        """removes overlapping faces of given Mesh objects from self (this is not a boolean)"""

        new = self.copy()
        for obj in args:
            tree = cKDTree(new.points)
            distances, idx = tree.query(obj.points)
            vertices = np.unique(idx[distances <= tolerance])

            if vertices.size:
                new = new.from_vertices(vertices, contained=True, exclude=True)

        self.indices = new.indices
        self.counts  = new.counts
        self.points  = new.points

        # reset components
        self.reset_cached_data()

    def symmetrical(
        self,
        pivot:         float = 0.0,
        axis:          int   = 0,
        tolerance:     float = 1e-6,
        check_topology       = True,
    ):
        """uv symmetry with pivot default to 0.5 in u"""
        return super().symmetrical(
            pivot          = pivot,
            axis           = axis,
            tolerance      = tolerance,
            check_topology = True,
            check_normals  = False,
        )

    def normalize(self, bbx_min=(0.0, 0.0), bbx_max=(1.0, 1.0), tolerance=1e-6):
        """brings UV maps within frame range (default 0-1)"""

        # find the uv offset and scale ratios
        uv_min    = self.points.min(axis=0)
        uv_max    = self.points.max(axis=0)
        uv_scale  = uv_max - uv_min
        bbx_min   = np.asarray(bbx_min)
        bbx_max   = np.asarray(bbx_max)
        bbx_scale = bbx_max - bbx_min

        # create a mask to handle div by zero when uv_scale is 0
        mask      = uv_scale > tolerance
        uv_border = np.where(self.points == uv_max)

        # move to 0,0
        self.points[:, mask] -= uv_min[mask]

        # normalize
        self.points[:, mask] *= bbx_scale[mask] / uv_scale[mask]

        # fix zero scale points which should be at 1.0
        self.points[:, ~mask] %= 1.0
        self.points[uv_border] = bbx_max[uv_border[1]]

        # offset to frame_min
        self.points += bbx_min

    def pack(self, resolution=1024, padding=2, rotations=4):
        """
        Pack UV islands into [0,1]^2 using rasterized bitmap skyline algorithm.

        Parameters
        ----------
        resolution : int
            Bitmap resolution (pixels per UV unit). Higher = more precise.
        padding : int
            Pixel padding between islands.
        rotations : int
            Number of rotation candidates in [0deg, 180deg). Set 1 to disable.
        """
        from cgmath.geometry.pack import pack_islands

        pack_islands(self, resolution=resolution, padding=padding, rotations=rotations)

    # ------------------------------ RASTERIZATION ------------------------------- #

    @property
    def antialias(self):
        return self._antialias

    @antialias.setter
    def antialias(self, antialias: bool = False):
        self._antialias = antialias

    @property
    def renderer(self):
        # default to cv2
        if self._renderer is None:
            self.renderer = RenderMethod.CV2
        return self._renderer

    @renderer.setter
    def renderer(self, method: RenderMethod | str = RenderMethod.CV2):
        # fallback to skimage if cv2 not available
        method = RenderMethod(method)

        if method == RenderMethod.CV2:
            if cv2 is None:
                warnings.warn("cv2 not available, defaulting to skimage")
                self.renderer = RenderMethod.SKIMAGE
            else:
                self._renderer = RenderMethod.CV2

        elif skimage is None:
            raise RuntimeError(
                "skimage not found, scikit-image or opencv required to rasterize uvs"
            )

        else:
            self._renderer = RenderMethod.SKIMAGE

    @property
    def resolution(self) -> np.ndarray:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: Tuple[int, int]) -> None:
        """sets the render buffer resolution"""
        if isinstance(resolution, int):
            resolution = np.array([resolution, resolution], dtype=int)
        else:
            resolution = np.array(resolution, dtype=int)

        if not np.allclose(self._resolution, resolution):
            self._resolution = resolution
            self._buffer     = None

    @property
    def buffer(self) -> None:
        """returns the render buffer"""
        if self._buffer is None:
            self._buffer = np.zeros(
                (self._resolution[1], self._resolution[0], 3), np.uint8
            )

        return self._buffer

    @buffer.setter
    def buffer(self, buffer) -> None:
        """sets _buffer"""
        self._buffer     = np.asarray(buffer, dtype=np.uint8)
        self._resolution = np.array([self._buffer.shape[1], self._buffer.shape[0]])

    def buffer_from_file(self, fname: str) -> None:
        """loads an image to the render buffer from a file"""

        fname = os.path.expanduser(fname)

        if self.renderer == RenderMethod.CV2:
            buffer      = cv2.imread(fname)
            buffer      = cv2.flip(buffer, 0)
            self.buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)

        elif self.renderer == RenderMethod.SKIMAGE:
            buffer      = skimage.io.imread(fname)
            self.buffer = np.flipud(buffer)
        else:
            raise RuntimeError(
                "skimage or opencv required to load image to buffer, neither found"
            )

    def clear_buffer(self, color: Tuple[int, int, int] = (0, 0, 0)) -> None:
        """clears the render buffer"""
        self.buffer[:] = color

    def _process_draw_arguments(
        self,
        count:        int,
        indices:      None   | np.ndarray           = None,
        color:        None   | Tuple[int, int, int] = None,
        invert:       bool                          = False,
        mask:         UVData | np.ndarray | None    = None,
        clear_buffer: bool                          = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocessor for common draw arguments
        """

        # overlap mask
        # if we're given a mask, make sure the resolution matches
        if mask is not None:
            if isinstance(mask, UVData):
                mask = mask.buffer.copy()
            elif isinstance(mask, np.ndarray):
                mask = mask.copy()

            rez = mask.shape[:2][::-1]
            if not np.allclose(self.resolution, rez):
                self.resolution = rez

        # if no indices given, default to all faces
        if indices is None:
            if invert:
                return
            indices = np.arange(count)
        else:
            indices = np.array(indices)

        # if invert is True, invert the face selection
        if invert:
            negative          = np.ones(count, dtype=bool)
            negative[indices] = False
            indices           = np.where(negative)[0]

        # if no color is specified, use encoded indices
        if color is None:
            color = self.encode_rgb(indices)

        elif not isinstance(color, np.ndarray):
            color = np.tile(color, (indices.size, 1))

        else:
            color = np.array(color, dtype=np.intp)

        # reset the buffer pixel counters
        self._pixel_counts   = np.zeros(count, dtype=int)
        self._pixel_overlaps = np.zeros(count, dtype=int)
        self._pixel_ratios   = np.zeros(count, dtype=float)

        # clear buffer
        if clear_buffer:
            self.clear_buffer()

        return indices, color, mask

    @property
    def pixel_counts(self):
        return self._pixel_counts

    @property
    def pixel_overlaps(self):
        return self._pixel_overlaps

    @property
    def pixel_ratios(self):
        return self._pixel_ratios

    def draw_mask(
        self,
        indices: None | np.ndarray = None,
        invert:  bool              = False,
        contour: bool              = False,
    ) -> None:
        """draws a white mask"""

        # always clear the buffer when drawing a mask
        self.clear_buffer()
        self.draw_faces(indices=indices, invert=invert, color=(255, 255, 255))

        if not contour and indices is not None:
            self.draw_faces(
                indices      = indices,
                invert       = (not invert),
                color        = (0, 0, 0),
                clear_buffer = False,
            )

    def draw_faces(
        self,
        mask:         UVData               | np.ndarray | None = None,
        indices:      None                 | np.ndarray        = None,
        invert:       bool                                     = False,
        color:        Tuple[int, int, int] | np.ndarray | None = (255, 255, 255),
        tolerance:    float                                    = 0.0,
        clear_buffer: bool                                     = False,
    ) -> np.ndarray | None:
        """draws faces"""

        # preprocess common arguments
        indices, color, mask = self._process_draw_arguments(
            count        = self.counts.size,
            indices      = indices,
            color        = color,
            invert       = invert,
            mask         = mask,
            clear_buffer = clear_buffer,
        )

        # build hole lookup if hole metadata is present
        hole_vi_map = {}
        if (
            self.hole_faces is not None
            and self.hole_counts is not None
            and self.hole_indices is not None
            and len(self.hole_faces) > 0
        ):
            offset = 0
            for hi in range(len(self.hole_faces)):
                fi  = int(self.hole_faces[hi])
                hc  = int(self.hole_counts[hi])
                hvs = set(int(v) for v in self.hole_indices[offset : offset + hc])
                hole_vi_map.setdefault(fi, []).append(hvs)
                offset += hc

        # define the polygons
        vertex_indices = self.f2v[indices]
        points         = np.round(self.points * (self.resolution - 1)).astype(int)
        polygons       = points[vertex_indices]

        for i, p in enumerate(polygons):
            face_vi = vertex_indices[i]
            valid   = face_vi >= 0
            face_vi = face_vi[valid]
            p       = p[valid]

            # skip degenerate polygons
            if p.shape[0] < 3:
                continue

            # define the sprite buffer
            bbx_min, bbx_max = p.min(axis=0), p.max(axis=0)
            bbx = bbx_max - bbx_min + 1
            p   = p - bbx_min

            sprite = np.zeros((bbx[1], bbx[0]), dtype=np.uint8)

            # split outer/hole contours if this face has holes
            face_idx       = indices[i]
            has_face_holes = face_idx in hole_vi_map
            if has_face_holes:
                all_hole_vis = set()
                for hvs in hole_vi_map[face_idx]:
                    all_hole_vis.update(hvs)
                outer_mask    = np.array([int(v) not in all_hole_vis for v in face_vi])
                outer_p       = p[outer_mask]
                hole_contours = []
                for hvs in hole_vi_map[face_idx]:
                    hmask = np.array([int(v) in hvs for v in face_vi])
                    hole_contours.append(p[hmask])

            # --- opencv --- #
            if self.renderer == RenderMethod.CV2:
                aa = cv2.LINE_AA if self.antialias is True else cv2.LINE_8

                if has_face_holes:
                    contours = [outer_p] + hole_contours
                    cv2.fillPoly(sprite, contours, 255)
                    if self.antialias:
                        sprite_aa = np.zeros((bbx[1], bbx[0]), dtype=np.uint8)
                        for c in contours:
                            cv2.polylines(sprite_aa, [c], True, 255, 1, aa)
                        composite_sprite_aa(sprite_aa, sprite)
                else:
                    cv2.fillPoly(sprite, [p], 255)
                    if self.antialias:
                        sprite_aa = np.zeros((bbx[1], bbx[0]), dtype=np.uint8)
                        cv2.polylines(sprite_aa, [p], True, 255, 1, aa)
                        composite_sprite_aa(sprite_aa, sprite)

            # --- skimage --- #
            else:
                if has_face_holes:
                    # fill outer
                    rr, cc = skimage.draw.polygon(
                        outer_p[:, 1], outer_p[:, 0], sprite.shape
                    )
                    sprite[rr, cc] = 255
                    # subtract holes
                    for hp in hole_contours:
                        rr, cc = skimage.draw.polygon(hp[:, 1], hp[:, 0], sprite.shape)
                        sprite[rr, cc] = 0
                    # contour lines
                    all_contours = [outer_p] + hole_contours
                    if not self.antialias:
                        for c in all_contours:
                            for j in range(c.shape[0]):
                                edge = np.roll(c, j, axis=0)[:2]
                                rr, cc = skimage.draw.line(*edge.ravel()[::-1])
                                sprite[rr, cc] = 255
                    else:
                        for c in all_contours:
                            for j in range(c.shape[0]):
                                sprite_aa = np.zeros((bbx[1], bbx[0]), dtype=np.uint8)
                                edge      = np.roll(c, j, axis=0)[:2]
                                rr, cc, aa = skimage.draw.line_aa(*edge.ravel()[::-1])
                                sprite_aa[rr, cc] = (aa * 255).astype(np.uint8)
                                composite_sprite_aa(sprite_aa, sprite)
                else:
                    rr, cc = skimage.draw.polygon(p[:, 1], p[:, 0], sprite.shape)
                    sprite[rr, cc] = 255
                    if not self.antialias:
                        for j in range(p.shape[0]):
                            edge = np.roll(p, j, axis=0)[:2]
                            rr, cc = skimage.draw.line(*edge.ravel()[::-1])
                            sprite[rr, cc] = 255
                    else:
                        for j in range(p.shape[0]):
                            sprite_aa = np.zeros((bbx[1], bbx[0]), dtype=np.uint8)
                            edge      = np.roll(p, j, axis=0)[:2]
                            rr, cc, aa = skimage.draw.line_aa(*edge.ravel()[::-1])
                            sprite_aa[rr, cc] = (aa * 255).astype(np.uint8)
                            composite_sprite_aa(sprite_aa, sprite)

            # composite the sprite
            count, overlap, ratio = composite_sprite(
                sprite, self.buffer, mask, bbx_min, color[i]
            )
            self._pixel_counts[indices[i]]   = count
            self._pixel_overlaps[indices[i]] = overlap
            self._pixel_ratios[indices[i]]   = ratio

        # if a mask was given, return the overlap ratios
        if mask is not None:
            tolerance = max(0, min(tolerance, 1))
            if tolerance == 0.0:
                ratios = self.pixel_ratios > 0.0
            else:
                ratios = self.pixel_ratios >= tolerance

            return np.where(ratios)[0]

    def draw_edges(
        self,
        mask:         UVData               | np.ndarray | None = None,
        indices:      None                 | np.ndarray        = None,
        invert:       bool                                     = False,
        color:        Tuple[int, int, int] | np.ndarray | None = (255, 255, 255),
        thickness:    int                                      = 1,
        tolerance:    float                                    = 0.0,
        clear_buffer: bool                                     = False,
    ) -> np.ndarray | None:
        """draws edges"""

        # preprocess common arguments
        indices, color, mask = self._process_draw_arguments(
            count        = self.e2v.shape[0],
            indices      = indices,
            color        = color,
            invert       = invert,
            mask         = mask,
            clear_buffer = clear_buffer,
        )

        # define the edges
        vertex_indices = self.e2v[indices]
        edges          = self.points[vertex_indices]
        edges          = np.round(edges * (self.resolution - 1)).astype(int)

        for i, p in enumerate(edges):
            # define the sprite buffer
            bbx_min, bbx_max = p.min(axis=0), p.max(axis=0)
            bbx_min -= thickness - 1
            bbx_max += thickness - 1
            bbx = bbx_max - bbx_min + 1
            p   = p - bbx_min

            sprite = np.zeros((bbx[1], bbx[0]), dtype=np.uint8)

            # --- opencv --- #
            if self.renderer == RenderMethod.CV2:
                aa = cv2.LINE_AA if self.antialias is True else cv2.LINE_8
                cv2.line(sprite, tuple(p[0]), tuple(p[1]), 255, thickness, lineType=aa)

            # --- skimage --- #
            else:
                if thickness > 1:
                    warnings.warn(
                        "line width > 1 not supported with skimage, defaulting to 1"
                    )

                rr, cc = skimage.draw.line(p[0][1], p[0][0], p[1][1], p[1][0])
                sprite[rr, cc] = 255

                if self.antialias:
                    sprite_aa = np.zeros((bbx[1], bbx[0]), dtype=np.uint8)

                    rr, cc, aa = skimage.draw.line_aa(
                        p[0][1], p[0][0], p[1][1], p[1][0]
                    )
                    sprite_aa[rr, cc] = (aa * 255).astype(np.uint8)
                    composite_sprite_aa(sprite_aa, sprite)

            # composite the sprite
            count, overlap, ratio = composite_sprite(
                sprite, self.buffer, mask, bbx_min, color[i]
            )
            self._pixel_counts[indices[i]]   = count
            self._pixel_overlaps[indices[i]] = overlap
            self._pixel_ratios[indices[i]]   = ratio

        # if a mask was given, return the overlap ratios
        if mask is not None:
            tolerance = max(0, min(tolerance, 1))
            if tolerance == 0.0:
                ratios = self.pixel_ratios > 0.0
            else:
                ratios = self.pixel_ratios >= tolerance

            return np.where(ratios)[0]

    def draw_points(
        self,
        mask:         UVData               | np.ndarray | None = None,
        indices:      None                 | np.ndarray        = None,
        invert:       bool                                     = False,
        color:        Tuple[int, int, int] | np.ndarray | None = (255, 255, 255),
        radius:       int                                      = 1,
        tolerance:    float                                    = 0.0,
        clear_buffer: bool                                     = False,
    ) -> np.ndarray | None:
        # preprocess common arguments
        indices, color, mask = self._process_draw_arguments(
            count        = self.points.shape[0],
            indices      = indices,
            color        = color,
            invert       = invert,
            mask         = mask,
            clear_buffer = clear_buffer,
        )

        # define the points
        points = self.points[indices]
        points = np.round(points * (self.resolution - 1)).astype(int)

        # define the sprite buffer and draw the sprite
        diameter = 2 * radius - 1
        center   = (radius - 1, radius - 1)
        sprite   = np.zeros((diameter, diameter), dtype=np.uint8)

        # --- opencv --- #
        if self.renderer == RenderMethod.CV2:
            aa = cv2.LINE_AA if self.antialias is True else cv2.LINE_8
            cv2.circle(sprite, center, radius - 1, 255, -1, aa)

        # --- skimage --- #
        else:
            rr, cc = skimage.draw.disk(center, radius - self.antialias)
            sprite[rr, cc] = 255

            if self.antialias:
                sprite_aa = np.zeros((diameter, diameter), dtype=np.uint8)
                rr, cc, aa = skimage.draw.circle_perimeter_aa(
                    center[0], center[1], radius - 1
                )
                sprite_aa[rr, cc] = (aa * 255).astype(np.uint8)
                composite_sprite_aa(sprite_aa, sprite)

        # copy the sprite over points
        for i, p in enumerate(points):
            # composite the sprite
            count, overlap, ratio = composite_sprite(
                sprite, self.buffer, mask, p - radius + 1, color[i]
            )
            self._pixel_counts[indices[i]]   = count
            self._pixel_overlaps[indices[i]] = overlap
            self._pixel_ratios[indices[i]]   = ratio

        # if a mask was given, return the overlap ratios
        if mask is not None:
            tolerance = max(0, min(tolerance, 1))
            if tolerance == 0.0:
                ratios = self.pixel_ratios > 0.0
            else:
                ratios = self.pixel_ratios >= tolerance

            return np.where(ratios)[0]

    def to_image(self) -> Image:
        """creates a PIL Image object"""
        if Image is None:
            raise RuntimeError("PIL required to generate PIL.Image data")

        image = Image.fromarray(self.buffer)
        return ImageOps.flip(image)

    def imwrite(self, fname: str) -> str:
        """writes a png file"""

        fname = os.path.expanduser(fname)
        if not fname.lower().endswith(".png"):
            fname += ".png"
        img = self.to_image()
        img.save(fname)

        return fname

    def imshow(self) -> None:
        """shows the image in a window"""
        if cv2 is None:
            raise RuntimeError("opencv required to show images")

        buffer = cv2.flip(self.buffer, 0)
        buffer = cv2.cvtColor(buffer, cv2.COLOR_BGR2RGB)
        cv2.imshow(self.name, buffer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_overlap_faces(
        self,
        uvdata:     "UVData",
        indices:    Union[None, np.ndarray] = None,
        resolution: int                     = 2048,
        tolerance:  float                   = 0.0,
    ):
        """returns the self overlapping face indices over the given given UVData object"""

        # issue a deprecation warning
        text = """
        Method `get_overlap_faces` is being deprecated.
        To compute a face overlap do the following:

        # rasterize the mask
        uv_mask.draw_mask(indices=indices)
        overlap = uv_target.draw_faces(mask=uv_mask, tolerance=0.5)
        """
        warnings.warn(text, DeprecationWarning)

        # rasterize the mask
        uvdata.resolution = resolution
        uvdata.draw_mask(indices=indices)

        # rasterize the faces
        self.resolution = resolution
        return self.draw_faces(mask=uvdata, tolerance=tolerance)

    # -------------------------------- IMAGE I/O --------------------------------- #

    def get_minimum_resolution(self, pixels: int = 3) -> np.ndarray:
        """returns the minumum resolution to fit n pixels in a given bounding box"""

        min_bbx, max_bbx = self.get_extent(per_face=True)
        bbx = max_bbx - min_bbx
        x   = int(np.ceil(pixels / bbx[:, 0].min()))
        y   = int(np.ceil(pixels / bbx[:, 1].min()))

        x   = max([x, 1])
        y   = max([y, 1])

        return np.array((x, y))

    @staticmethod
    def _erode(image: Any, steps: int = 1, kernel: np.ndarray | int = 1) -> Any:
        """erodes (shrinks) image bordering black pixels"""

        # if given PIL image, convert to np.ndarray (convert back to PIL later)
        if image is None:
            raise ValueError("image must be np.ndarray or Image, got None.")

        is_array = isinstance(image, np.ndarray)
        if not is_array:
            if Image is None:
                raise RuntimeError("PIL required to handle PIL.Image data")
            image = np.array(image)

        # generate the kernel if given as an integer (star shaped)
        if isinstance(kernel, int):
            size   = kernel
            size   = (size + 1) * 2 - 1
            kernel = np.zeros((size, size), dtype=int)
            center_index                       = size // 2
            kernel[center_index, center_index] = 0
            kernel[center_index, :]            = 1
            kernel[:, center_index]            = 1

        # convolve the image with the kernel
        for _ in range(steps):
            # create a convolution mask for non black pixels
            mask      = np.any(image != [0, 0, 0], axis=-1).astype(int)
            convolved = convolve(mask, kernel, mode="constant", cval=0.0)
            eroded    = convolved == np.sum(kernel)

            eroded_image         = np.zeros_like(image)
            eroded_image[eroded] = image[eroded]
            image                = eroded_image

        if not is_array:
            return Image.fromarray(image)

        return image

    def erode(self, steps: int = 1, kernel: np.ndarray | int = 1) -> None:
        """erodes (shrinks) the buffer's bordering black pixels"""
        self.buffer = self._erode(self.buffer, steps=steps, kernel=kernel)

    @staticmethod
    def _convolve(image: Any, steps: int = 1, kernel: np.ndarray | int = 1) -> Any:
        """convolves (grows) an image over black pixels"""

        # if given PIL image, convert to np.ndarray (convert back to PIL later)
        if image is None:
            raise ValueError("image must be np.ndarray or Image, got None.")

        is_array = isinstance(image, np.ndarray)
        if not is_array:
            if Image is None:
                raise RuntimeError("PIL required to handle PIL.Image data")
            image = np.array(image)

        # generate the kernel if given as an integer (doughnut shaped)
        if isinstance(kernel, int):
            size   = kernel
            size   = (size + 1) * 2 - 1
            kernel = np.ones((size, size), dtype=int)
            center_index                       = size // 2
            kernel[center_index, center_index] = 0

        # convolve the image with the kernel
        for _ in range(steps):
            # create a convolution mask for non black pixels
            mask      = np.any(image != [0, 0, 0], axis=-1).astype(int)
            convolved = convolve(mask, kernel, mode="constant", cval=0.0)

            # identify black and non-black pixels
            black     = np.all(image == [0, 0, 0], axis=-1)
            non_black = np.any(image != [0, 0, 0], axis=-1)

            # find black pixels that are neighboring non-black pixels
            indices      = np.where((black) & (convolved != 0))
            black_pixels = np.array(indices).T
            if black_pixels.size == 0:
                break
            non_black_pixels = np.array(np.where(non_black)).T

            # use a kd tree to find the closest non black value
            tree = cKDTree(non_black_pixels)
            _, idx = tree.query(black_pixels)

            # set the closest indices back into the array
            image[indices] = image[
                non_black_pixels[idx][:, 0], non_black_pixels[idx][:, 1]
            ]

        if not is_array:
            return Image.fromarray(image)

        return image

    def convolve(self, steps: int = 1, kernel: np.ndarray | int = 1) -> None:
        """convolves (grows) the buffer over black pixels"""
        self.buffer = self._convolve(self.buffer, steps=steps, kernel=kernel)

    @staticmethod
    def encode_rgb(i: int | np.ndarray) -> Tuple[int, int, int] | np.ndarray:
        max_index = 256**3 - 2
        i         = np.asarray(i)
        if np.any(i > max_index):
            raise RuntimeError(f"Exceeded max integer {max_index}")

        r = ((i + 1) // (256 * 256)) % 256
        g = ((i + 1) // 256) % 256
        b = (i + 1) % 256

        return np.column_stack((r, g, b)).astype(np.intp)

    @staticmethod
    def decode_rgb(rgb: Tuple[int, int, int] | np.ndarray) -> int | np.ndarray:
        rgb = np.asarray(rgb, dtype=int)
        if rgb.ndim == 1:
            rgb = rgb[None, :]

        if np.any(np.sum(rgb == 0, axis=1) == 3):
            raise RuntimeError("Cannot decode rgb 0,0,0")

        return rgb[:, 0] * 256**2 + rgb[:, 1] * 256 + rgb[:, 2] - 1

    # ------------------------------ HOLE PATCHING ------------------------------- #

    @property
    def has_holes(self) -> bool:
        """returns True if the UVData has holes"""
        return self.get_holes().size > 0

    def get_holes(self) -> np.ndarray:
        """returns the missing face indices (aka holes)"""
        return np.where(self.counts == 0)[0]

    def fill_holes(self, counts: np.ndarray) -> bool:
        """fills empty faces with the expected face counts and default uvs"""

        indices = self.get_holes()
        if indices.size == 0:
            return True

        # make a geometry buffer
        geometry = self.geometry.copy()

        # for each polygon type (tri, quad, n5, n6, etc...)
        if self.geometry.size:
            offset_uv, _ = self.get_extent()
        else:
            offset_uv = np.zeros(2)

        for i, n in enumerate(np.unique(counts[counts > 0])):
            missing = np.where(counts[indices] == n)[0]

            # do we have any holes for this polygon type?
            if missing.size > 0:
                # expand geometry buffer if needed
                if geometry.shape[1] < n:
                    buffer                         = np.ones((geometry.shape[0], n), dtype=np.int32) * -1
                    buffer[:, : geometry.shape[1]] = geometry
                    geometry                       = buffer

                # generate a nice shape outside of 0-1 range to avoid overlap
                angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                if n == 3:
                    angles *= np.pi / 2
                elif n == 4:
                    angles *= np.pi / 4

                x     = 0.5 + 0.5 * np.cos(angles) - (i + 1) + offset_uv[0]
                y     = 0.5 + 0.5 * np.sin(angles) + offset_uv[1] - 1
                shape = np.column_stack((x, y))

                # append indices and new points
                tiles                          = np.tile(shape, (missing.size, 1))
                new_indices                    = np.arange(n * missing.size) + self.points.shape[0]
                geometry[indices[missing], :n] = new_indices.reshape(missing.size, n)

                self.points = np.concatenate((self.points, tiles))

        # update self counts and indices
        self.indices, self.counts = matrix_to_stream(geometry)
        self.reset_cached_data()

        # return True if uv face counts match expected counts
        return np.allclose(self.counts, counts)

    def patch_holes(self, uv_data: "UVData") -> bool:
        """combines the given UVData object into this one"""

        if not self.has_holes:
            raise RuntimeError("No holes to patch")

        # make sure the two objects complete each other with no overlap
        test = (self.counts > 0).astype(int) + (uv_data.counts > 0).astype(int)
        if np.any(test > 1):
            return False

        # find the holes to patch
        geometry        = self.geometry.copy()
        points          = self.points.copy()
        missing_indices = self.get_holes()

        # isolate the corresponding geometry patch
        missing_geometry = uv_data.geometry[missing_indices]
        mask             = missing_geometry == -1

        # offset the missing geometry indices by point count
        missing_geometry += points.shape[0]
        missing_geometry[mask] = -1

        # insert the missing geometry back into the original
        geometry[missing_indices] = missing_geometry

        # concatenate the points
        self.points = np.concatenate([points, uv_data.points])
        self.indices, self.counts = matrix_to_stream(geometry)

        self.reset_cached_data()

        return True

    # ----------------------------------- GLB ------------------------------------ #

    @classmethod
    def load_glb(cls, filename: str) -> List["UVData"]:
        """a basic glb file reader to UVData"""

        if trimesh is None:
            raise ImportError("trimesh is not installed")

        filename = os.path.expanduser(filename)
        scene    = trimesh.load(filename)

        uv_lists = []
        for name, data in scene.geometry.items():
            uv_lists.append(None)

            if data.visual.uv is not None:
                points       = np.array(data.visual.uv)
                indices      = np.array(data.visual.mesh.faces)
                counts       = np.ones(indices.shape[0], dtype=int) * 3
                indices      = indices.ravel()
                uv_lists[-1] = cls(points=points, indices=indices, counts=counts)

        return uv_lists

    # ----------------------------------- FBX ------------------------------------ #

    @classmethod
    def load_fbx(
        cls, filename: str, name: str | None = None, channel: int = 0
    ) -> "UVData":
        """returns a single UV channel from an fbx file (first mesh / channel 0 by default)"""
        uv_list = UVList.load_fbx(filename, name=name)
        if channel >= len(uv_list):
            raise IndexError(
                f"UV channel {channel} not found; mesh has {len(uv_list)} channel(s)"
            )
        return uv_list[channel]

    # --------------------------------- USD I/O ---------------------------------- #

    @classmethod
    def from_prim(cls, prim: Any) -> List["UVData"]:
        """Construct a list of data objects from a mesh prim."""

        def iter_uv_attrs() -> Tuple[Any]:
            """Iterats UV attribute pairs in a mesh prim."""
            for attr in prim.GetAuthoredAttributes():
                attr_name = attr.GetName()
                if attr_name.startswith(UV_ATTR_PREFIX):
                    i = attr_name.replace(UV_ATTR_PREFIX, "")
                    if not i or i.isdigit():
                        id_attr_name = f"{attr_name}:indices"
                        yield attr, prim.GetAttribute(id_attr_name)

        mesh_api    = pxr().UsdGeom.Mesh(prim)
        vert_counts = mesh_api.GetFaceVertexCountsAttr().Get()

        data_list   = []
        for i, (uv_attr, uv_ids_attr) in enumerate(iter_uv_attrs()):
            custom_data = uv_attr.GetCustomDataByKey("Maya") or {}
            uv_set      = custom_data.get("name", f"uv_set{i}")
            uv_ids      = uv_ids_attr.Get()
            data = cls(
                name    = uv_set,
                indices = np.array(uv_ids),
                counts  = np.array(vert_counts),
                points  = np.array(uv_attr.Get()),
            )

            # rare case where indices are not indexed, assume mesh is facetted
            # TODO change iter_uv_attrs to test if primvar "IsIndexed"
            if uv_ids is None:
                data.indices = np.arange(data.points.shape[0])

            data_list.append(data)

        return UVList(data_list)

    def to_prim(self, uv_set_id: int, prim: Any) -> None:
        """Writes UV data into a mesh prim.

        Args:
            uv_set_id: An uv set index to write the UV data under.
            prim: A mesh prim to write the UV data to.
        """
        Sdf      = pxr().Sdf
        var_name = UV_ATTR_PREFIX if uv_set_id == 0 else f"{UV_ATTR_PREFIX}{uv_set_id}"
        attr = prim.CreateAttribute(
            var_name,
            Sdf.ValueTypeNames.TexCoord2fArray,
            False,
            Sdf.VariabilityVarying,
        )
        attr.Set(self.points)
        attr.SetCustomDataByKey("Maya", {"name": self.name})
        attr.SetMetadata("interpolation", "faceVarying")
        attr = prim.CreateAttribute(
            f"{var_name}:indices",
            Sdf.ValueTypeNames.IntArray,
            False,
            Sdf.VariabilityVarying,
        )
        attr.Set(self.indices)


class MeshList(DataList):
    DATA_LIST_CLASS = MeshData

    @classmethod
    def load_fbx(cls, filename: str) -> "MeshList":
        """loads all meshes from an fbx file into a MeshList"""
        return cls([mesh_data for mesh_data, _ in load_fbx(filename)])

    # --------- vectorized properties and methods --------- #
    def merge(self):
        """applies merge to all elements"""
        [x.merge() for x in self]

    def to_identity(self):
        """sets all contained MeshData objects to identity"""
        for obj in self:
            obj.to_identity()

    def fix_symmetry(
        self,
        pivot:     float              = 0.0,
        axis:      int                = 0,
        side:      float              = 1.0,
        combined:  bool               = True,
        tolerance: Union[None, float] = None,
    ) -> np.ndarray:
        """
        attempts a symmetry fix on each element or combined.
        returns a bool array indicating success per mesh.
        """

        # combined symmetry
        if combined:
            face_counts = np.array([x.face_count for x in self])
            face_sums   = np.cumsum(face_counts)
            face_sums   = np.insert(face_sums, 0, 0)
            face_starts = face_sums[:-1]
            face_ends   = face_sums[1:]

            obj = self[0].copy()
            obj.to_identity()
            obj.union(*self[1:])

            if obj.fix_symmetry(pivot=pivot, axis=axis, side=side, tolerance=tolerance):
                for i in range(face_starts.size):
                    indices   = np.arange(face_starts[i], face_ends[i])
                    mesh      = obj.from_faces(indices)
                    mesh.name = self[i].name

                    # bring back points to local space
                    points = mesh.points4 - self[i].matrix[3]
                    mesh.points = np.dot(points, self[i].get_inverse_matrix().T)[
                        :, : mesh.points.shape[1]
                    ]
                    self[i].points = mesh.points

                return np.ones(face_counts.size, dtype=bool)
            else:
                return np.zeros(face_counts.size, dtype=bool)

        # individual symmetry
        result = []
        for obj in self:
            result.append(
                obj.fix_symmetry(pivot=pivot, axis=axis, side=side, tolerance=tolerance)
            )

        return np.array(result)


class UVList(MeshList):
    DATA_LIST_CLASS = UVData

    @classmethod
    def load_fbx(cls, filename: str, name: str | None = None) -> "UVList":
        """loads all UV channels for one mesh from an fbx file (first mesh if name is None)"""
        data = load_fbx(filename)
        if not data:
            raise RuntimeError("No meshes found in FBX file")
        if name is None:
            return data[0][1]
        for mesh_data, uv_list in data:
            if mesh_data.name == name:
                return uv_list
        available = [m.name for m, _ in data]
        raise ValueError(f"Mesh '{name}' not found; available: {available}")

    # --------- vectorized properties and methods --------- #

    def get_minimum_resolution(self):
        """computes the minimum resolution for the entire list"""
        resolutons = [[1, 1]]
        for obj in self:
            r = obj.get_minimum_resolution()
            resolutons.append(r)
        return np.max(resolutons, axis=0)

    def fix_symmetry(
        self,
        pivot:     float              = 0.5,
        axis:      int                = 0,
        side:      float              = 1.0,
        combined:  bool               = True,
        tolerance: Union[None, float] = None,
    ) -> np.ndarray:
        return super().fix_symmetry(
            self,
            pivot     = pivot,
            axis      = axis,
            side      = side,
            combined  = combined,
            tolerance = tolerance,
        )

    def triangulate(self, rules: Union[np.ndarray, TriangulateRules]) -> None:
        """applies the triangulation rules to all UVData in a UVList"""
        for obj in self:
            obj.triangulate(rules=rules)

    def from_vertices(self, *args, **kwarg) -> "UVList":
        """returns a new UVList object from the given vertex indices or rules"""
        new = []
        for obj in self:
            new.append(obj.from_vertices(*args, **kwarg))
        return self.__class__(new)

    def from_faces(self, *args, **kwarg) -> "UVList":
        """returns a new UVList object from the given face indices"""
        new = []
        for obj in self:
            new.append(obj.from_faces(*args, **kwarg))
        return self.__class__(new)

    # ------------------------------- UV PACKING --------------------------------- #
    def pack(self, resolution=1024, padding=2, rotations=4):
        [
            x.pack(resolution=resolution, padding=padding, rotations=rotations)
            for x in self
        ]

    # ------------------------------ HOLE PATCHING ------------------------------- #

    @property
    def has_holes(self) -> np.ndarray:
        """returns True/False for each UVData if it has holes"""
        return np.array([x.has_holes for x in self])

    def fill_holes(self, counts: np.ndarray) -> np.ndarray:
        """fills empty faces with the expected face counts and default uvs"""
        return np.array([x.fill_holes(counts) for x in self])

    def merge_seams(self):
        """merges the UV seams"""
        [x.merge() for x in self]

    def defrag(self, keep_order: bool = True) -> None:
        """recombined fragmented UVData objects into complete UVData objects"""

        # only proceed if UVData is fragmented
        if np.any(self.has_holes):
            # build the uv hole puzzle pieces
            puzzle = np.zeros((len(self), self[0].counts.size), dtype=bool)
            for i, obj in enumerate(self):
                puzzle[i] = obj.counts > 0

            # solve the puzzle by finding combinations of rows leave no holes once combined
            solution  = []
            used_rows = set()

            while True:
                # find all the unused puzzle rows
                unused_rows = [i for i in range(len(self)) if i not in used_rows]

                # break if all the puzzle pieces were used
                if not unused_rows:
                    break

                # try to find combinations of rows that sum to num_cols
                found = False
                for r in range(1, len(unused_rows) + 1):
                    for combo in combinations(unused_rows, r):
                        if np.all(np.sum(puzzle[list(combo)], axis=0) == 1):
                            solution.append(list(combo))
                            used_rows.update(combo)
                            found = True
                            break

                    # stop combinations if a solution was found
                    if found:
                        break

                # break if no more solutions exist
                if not found:
                    break

            # add any remaining rows that couldn't be combined
            remaining_rows = [i for i in unused_rows if i not in used_rows]
            for row in remaining_rows:
                solution.append([row])

            # sort the solution to maintain the original order
            if keep_order:
                solution.sort(key=lambda x: x[0])

            # use the solution to patch the UVData objects
            new_list = []
            for combo in solution:
                new_list.append(self[combo[0]].copy())

                for i in combo[1:]:
                    new_list[-1].patch_holes(self[i])

            # replace the old list with the new
            self[:] = new_list


# ------------------------------------ UTILS ------------------------------------- #
def load_usd(file_path: str) -> list:
    """loads usd file and returns list of (MeshData, UVList) tuples"""

    if iter_prims is None:
        raise ImportError("pxr is not installed")

    file_path = os.path.expanduser(file_path)
    stage     = open_stage(file_path, no_cache=True)

    data = []
    for prim in iter_prims(stage, "Mesh"):
        mesh_data    = MeshData.from_prim(prim)
        uv_data_list = UVData.from_prim(prim)
        data.append((mesh_data, uv_data_list))

    return data


def load_glb(file_path: str, scale_factor: float = 100.0) -> list:
    """loads glb file and returns list of (MeshData, UVList) tuples"""

    if trimesh is None:
        raise ImportError("trimesh is not installed")

    file_path = os.path.expanduser(file_path)
    scene     = trimesh.load(file_path)

    data = []
    for name, geometry in scene.geometry.items():
        points  = np.array(geometry.vertices) * scale_factor
        indices = np.array(geometry.faces)
        counts  = np.ones(indices.shape[0], dtype=int) * 3

        mesh_data = MeshData(
            points  = points,
            indices = indices.ravel(),
            counts  = counts,
            name    = name,
        )

        uv_data_list = []
        # trimesh returns TextureVisuals (.uv) for textured meshes and
        # ColorVisuals (no .uv) for vertex-colored / material-less meshes.
        # Use getattr so the absent attribute is treated the same as no UVs.
        uv_attr = getattr(geometry.visual, "uv", None)
        if uv_attr is not None:
            uv_points = np.array(uv_attr)
            uv_data = UVData(
                points  = uv_points,
                indices = indices.ravel(),
                counts  = counts,
            )
            uv_data_list.append(uv_data)

        data.append((mesh_data, UVList(uv_data_list)))

    return data


def load_obj(file_path: str) -> list:
    """loads obj file and returns list of (MeshData, UVList) tuples"""

    file_path = os.path.expanduser(file_path)

    # global accumulators (obj indices are global across the whole file)
    all_points  = []
    all_uvs     = []
    all_normals = []

    # per-group accumulators
    groups  = []
    current = None

    def _new_group(name):
        g = {
            "name":        name,
            "vert_ids":    [],
            "uv_ids":      [],
            "normal_ids":  [],
            "vert_counts": [],
        }
        groups.append(g)
        return g

    with open(file_path, "r") as f:
        for line in f.readlines():
            if line.startswith("g "):
                tokens = line.split()
                name   = tokens[1] if len(tokens) > 1 else None
                if name == "default":
                    name = None
                current = _new_group(name)

            elif line.startswith("vn "):
                all_normals.append([float(x) for x in line.split()[1:]])

            elif line.startswith("vt "):
                all_uvs.append([float(x) for x in line.split()[1:]])

            elif line.startswith("v "):
                all_points.append([float(x) for x in line.split()[1:]])

            elif line.startswith("f "):
                if current is None:
                    current = _new_group(None)

                tokens       = line.split()[1:]
                face_verts   = []
                face_uvs     = []
                face_normals = []
                for tok in tokens:
                    parts = tok.split("/")
                    face_verts.append(int(parts[0]))
                    if len(parts) >= 2 and parts[1]:
                        face_uvs.append(int(parts[1]))
                    if len(parts) >= 3 and parts[2]:
                        face_normals.append(int(parts[2]))
                current["vert_ids"].extend(face_verts)
                current["uv_ids"].extend(face_uvs)
                current["normal_ids"].extend(face_normals)
                current["vert_counts"].append(len(face_verts))

    all_points  = np.array(all_points)
    all_uvs     = np.array(all_uvs) if all_uvs else None
    all_normals = np.array(all_normals) if all_normals else None

    data = []
    for group in groups:
        if not group["vert_ids"]:
            continue

        vert_ids    = np.array(group["vert_ids"]) - 1
        vert_counts = np.array(group["vert_counts"])

        # de-duplicate global points to a per-mesh point set
        used_points, point_remap = np.unique(vert_ids, return_inverse=True)
        mesh_points = all_points[used_points]

        # re-insert "Shape" marker, matching MeshData.load_obj behavior
        name = group["name"]
        if name is not None:
            match = re.search(r"\d+$", name)
            if match:
                index = match.start()
                name  = f"{name[:index]}Shape{name[index:]}"
            else:
                name = f"{name}Shape"

        mesh_data = MeshData(
            points  = mesh_points,
            indices = point_remap.astype(np.int64),
            counts  = vert_counts,
            name    = name,
        )

        if all_normals is not None and group["normal_ids"]:
            normal_ids = np.array(group["normal_ids"]) - 1
            used_normals, normal_remap = np.unique(normal_ids, return_inverse=True)
            mesh_data.normals        = all_normals[used_normals]
            mesh_data.normal_indices = normal_remap.astype(np.int64)

        uv_data_list = []
        if all_uvs is not None and group["uv_ids"]:
            uv_ids = np.array(group["uv_ids"]) - 1
            used_uvs, uv_remap = np.unique(uv_ids, return_inverse=True)
            uv_points = all_uvs[used_uvs]
            uv_data = UVData(
                points  = uv_points,
                indices = uv_remap.astype(np.int64),
                counts  = vert_counts,
            )
            uv_data_list.append(uv_data)

        data.append((mesh_data, UVList(uv_data_list)))

    return data


def save_obj(file_path: str, data: list) -> None:
    """writes a valid obj file from a list of (MeshData, UVList) tuples"""

    file_path = os.path.expanduser(file_path)
    with open(file_path, "w") as f:
        f.write("# OBJ file\n")

        mesh_offset = 0
        uv_offset   = 0
        for mesh_data, uv_data_list in data:
            name = mesh_data.name
            if name is not None:
                if bool(re.search("Shape[0-9]*$", name)):
                    name = "".join(name.rpartition("Shape")[::2])

                f.write(f"g {name}\n")

            # vertices
            f.write("s 1\n")
            for p in mesh_data.points:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")

            # uvs (assume only one uv set)
            for p in uv_data_list[0].points:
                f.write(f"vt {p[0]} {p[1]}\n")

            # faces
            for i, c in enumerate(mesh_data.counts):
                mesh_faces = [
                    str(x + mesh_offset)
                    for x in (mesh_data.geometry[i, :c] + 1).tolist()
                ]
                uv_faces = [
                    str(x + uv_offset)
                    for x in (uv_data_list[0].geometry[i, :c] + 1).tolist()
                ]
                vals = " ".join(f"{a}/{b}" for a, b in zip(mesh_faces, uv_faces))
                f.write(f"f {vals}\n")

            mesh_offset += mesh_data.points.shape[0]
            uv_offset   += uv_data_list[0].points.shape[0]

        f.write("\n")


def _walk_fbx_mesh_nodes(node, out):
    """Recursively collect FBX nodes whose attribute is a mesh."""
    attr = node.GetNodeAttribute()
    if attr is not None and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
        out.append(node)
    for i in range(node.GetChildCount()):
        _walk_fbx_mesh_nodes(node.GetChild(i), out)


def _fbx_control_points(mesh) -> np.ndarray:
    n      = mesh.GetControlPointsCount()
    points = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        cp           = mesh.GetControlPointAt(i)
        points[i, 0] = cp[0]
        points[i, 1] = cp[1]
        points[i, 2] = cp[2]
    return points


def _fbx_polygon_streams(mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (indices, counts) -- flat face-vertex stream + per-face vertex count."""
    n_polys = mesh.GetPolygonCount()
    counts  = np.empty(n_polys, dtype=np.int64)
    for i in range(n_polys):
        counts[i] = mesh.GetPolygonSize(i)

    indices = np.empty(int(counts.sum()), dtype=np.int64)
    offset  = 0
    for face_i in range(n_polys):
        sz = int(counts[face_i])
        for v in range(sz):
            indices[offset + v] = mesh.GetPolygonVertex(face_i, v)
        offset += sz
    return indices, counts


def _fbx_direct_array_to_ndarray(direct_array, dim: int) -> np.ndarray:
    """Copy an FbxLayerElementArray of FbxVector{2,3,4} into an (N, dim) ndarray."""
    n   = direct_array.GetCount()
    out = np.empty((n, dim), dtype=np.float64)
    for i in range(n):
        v = direct_array.GetAt(i)
        for d in range(dim):
            out[i, d] = v[d]
    return out


def _fbx_face_varying_indices(mesh, layer_element, counts: np.ndarray) -> np.ndarray:
    """Project an FBX layer element's mapping/reference modes into a face-varying
    index stream compatible with MeshData (length = sum(counts), each entry indexes
    the layer element's direct array)."""

    mapping     = layer_element.GetMappingMode()
    reference   = layer_element.GetReferenceMode()
    index_array = layer_element.GetIndexArray()

    EMap = fbx.FbxLayerElement.EMappingMode
    ERef = fbx.FbxLayerElement.EReferenceMode

    total   = int(counts.sum())
    indices = np.empty(total, dtype=np.int64)

    if mapping == EMap.eByPolygonVertex:
        if reference == ERef.eIndexToDirect:
            for i in range(total):
                indices[i] = index_array.GetAt(i)
        elif reference == ERef.eDirect:
            for i in range(total):
                indices[i] = i
        else:
            raise NotImplementedError(
                f"Unsupported FBX reference mode for ByPolygonVertex: {reference}"
            )

    elif mapping == EMap.eByControlPoint:
        n_polys = mesh.GetPolygonCount()
        offset  = 0
        for face_i in range(n_polys):
            sz = int(counts[face_i])
            for v in range(sz):
                cp = mesh.GetPolygonVertex(face_i, v)
                if reference == ERef.eIndexToDirect:
                    indices[offset + v] = index_array.GetAt(cp)
                elif reference == ERef.eDirect:
                    indices[offset + v] = cp
                else:
                    raise NotImplementedError(
                        f"Unsupported FBX reference mode for ByControlPoint: {reference}"
                    )
            offset += sz

    else:
        raise NotImplementedError(f"Unsupported FBX mapping mode: {mapping}")

    return indices


def _extract_fbx_uv_channels(mesh, counts: np.ndarray) -> List["UVData"]:
    """Build one UVData per UV element in the mesh."""
    uv_list = []
    for layer_i in range(mesh.GetElementUVCount()):
        uv_element = mesh.GetElementUV(layer_i)
        if uv_element is None:
            continue

        points  = _fbx_direct_array_to_ndarray(uv_element.GetDirectArray(), dim=2)
        indices = _fbx_face_varying_indices(mesh, uv_element, counts)
        name    = uv_element.GetName() or f"map{layer_i + 1}"

        uv_list.append(
            UVData(
                name    = name,
                indices = indices,
                counts  = counts.copy(),
                points  = points,
            )
        )
    return uv_list


def _extract_fbx_normals(
    mesh, counts: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (normals, normal_indices) face-varying, or (None, None) if not authored."""
    if mesh.GetElementNormalCount() == 0:
        return None, None

    normal_element = mesh.GetElementNormal(0)
    normals        = _fbx_direct_array_to_ndarray(normal_element.GetDirectArray(), dim=3)
    normal_indices = _fbx_face_varying_indices(mesh, normal_element, counts)
    return normals, normal_indices


def load_fbx(file_path: str) -> list:
    """loads fbx file and returns list of (MeshData, UVList) tuples"""

    if fbx is None:
        raise ImportError("Autodesk FBX Python SDK is not installed")

    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FBX file not found: {file_path}")

    manager = fbx.FbxManager.Create()
    try:
        ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
        manager.SetIOSettings(ios)

        importer = fbx.FbxImporter.Create(manager, "Importer")
        if not importer.Initialize(file_path, -1, manager.GetIOSettings()):
            error = importer.GetStatus().GetErrorString()
            importer.Destroy()
            raise RuntimeError(f"Failed to initialize FBX importer: {error}")

        scene = fbx.FbxScene.Create(manager, "ImportedScene")
        if not importer.Import(scene):
            error = importer.GetStatus().GetErrorString()
            importer.Destroy()
            raise RuntimeError(f"Failed to import FBX scene: {error}")
        importer.Destroy()

        mesh_nodes = []
        _walk_fbx_mesh_nodes(scene.GetRootNode(), mesh_nodes)

        data = []
        for node in mesh_nodes:
            mesh   = node.GetNodeAttribute()
            points = _fbx_control_points(mesh)
            indices, counts = _fbx_polygon_streams(mesh)
            normals, normal_indices = _extract_fbx_normals(mesh, counts)

            mesh_data = MeshData(
                points         = points,
                indices        = indices,
                counts         = counts,
                name           = node.GetName(),
                normals        = normals,
                normal_indices = normal_indices,
            )

            uv_data_list = _extract_fbx_uv_channels(mesh, counts)
            data.append((mesh_data, UVList(uv_data_list)))

        return data
    finally:
        manager.Destroy()


# --------------------------------- NORMAL HELPERS ---------------------------------- #


def _detect_hard_edges(mesh, angle_deg):
    """Return edge indices where adjacent face-normal angle exceeds threshold."""
    face_normals = mesh.get_face_normals()
    e2f          = mesh.e2f

    valid     = np.all(e2f >= 0, axis=1)
    valid_idx = np.where(valid)[0]

    n0 = face_normals[e2f[valid_idx, 0]]
    n1 = face_normals[e2f[valid_idx, 1]]

    dot    = np.einsum("ij,ij->i", n0, n1)
    dot    = np.clip(dot, -1.0, 1.0)
    angles = np.degrees(np.arccos(dot))

    return valid_idx[angles > angle_deg]


def _compute_shading_normals(
    mesh, normal_indices, num_normals, angle_weighted, area_weighted
):
    """Accumulate face normals into face-varying normal slots."""
    face_normals = mesh.get_face_normals(force=True)
    fv_normals   = np.repeat(face_normals, mesh.counts, axis=0)

    if area_weighted:
        areas = mesh.get_face_areas()
        fv_normals *= np.repeat(areas, mesh.counts)[:, None]

    if angle_weighted:
        fv_normals *= mesh.get_face_vertex_angles(force=True)[:, None]

    summed = np.zeros((num_normals, 3))
    np.add.at(summed, normal_indices, fv_normals)

    with np.errstate(divide="ignore", invalid="ignore"):
        mag     = np.einsum("...i,...i", summed, summed) ** 0.5
        normals = summed / mag[:, None]

    degenerate          = ~np.isfinite(normals).all(axis=1)
    normals[degenerate] = [0, 1, 0]

    return normals


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Per-row L2 normalization with a divide-by-zero guard."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def _cross_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row scalar 2D cross product (z-component of the 3D cross).

    Inputs are ``(N, 2)`` arrays; returns an ``(N,)`` array of signed
    scalars.  Used in the convexity check for 2D meshes (``UVData``).
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _score_quad_candidates(
    mesh:              MeshData,
    planarity_weight:  float,
    squareness_weight: float,
    diagonal_weight:   float,
    min_score:         float,
    require_convex:    bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-edge quad-merge scoring used by ``MeshData.get_quadrangulate_rules``.

    Returns a tuple ``(score, fa, fb, cand)`` where ``cand`` are the indices
    of candidate unique edges (interior edges between two triangles), ``fa``
    and ``fb`` are the two triangles incident to each candidate, and
    ``score`` is the per-candidate quality.  Edges that cannot be merged
    (border, n-gon neighbor, non-manifold, concave) receive ``-np.inf``.
    """

    counts = mesh.counts
    ue2f   = mesh.ue2f
    ue2v   = mesh.ue2v
    f2v    = mesh.f2v
    points = mesh.points

    # No interior edges at all (e.g. a single isolated face or a pure quad
    # mesh with no shared edges) - bail out before touching the second
    # column of ``ue2f``.
    if ue2f.shape[1] < 2:
        empty_cand = np.empty(0, dtype=int)
        return np.empty(0), np.empty(0, dtype=int), np.empty(0, dtype=int), empty_cand

    # 2D meshes (e.g. ``UVData``) cannot have meaningful face normals or
    # dihedrals - everything is on a single plane by construction.  We skip
    # the planarity term and use a 2D scalar cross product for convexity.
    is_2d        = points.shape[1] == 2
    face_normals = None if is_2d else mesh.get_face_normals()

    # Manifold gating: exactly 2 incident faces, both triangles.  ``ue2f``
    # rows can have more than 2 slots when the mesh is non-manifold; we
    # require exactly two non-negative entries to keep the merge well-defined.
    n_face_per_edge = np.sum(ue2f >= 0, axis=1)
    manifold        = n_face_per_edge == 2
    fa_all          = ue2f[:, 0]
    fb_all          = ue2f[:, 1]

    both_tri = np.zeros(ue2f.shape[0], dtype=bool)
    both_tri[manifold] = (counts[fa_all[manifold]] == 3) & (
        counts[fb_all[manifold]] == 3
    )
    cand = np.where(both_tri)[0]
    if cand.size == 0:
        return np.empty(0), np.empty(0, dtype=int), np.empty(0, dtype=int), cand

    fa = fa_all[cand]
    fb = fb_all[cand]
    va = ue2v[cand, 0]
    vb = ue2v[cand, 1]

    # For each triangle, find the vertex NOT on the shared edge.  ``f2v`` may
    # be padded with -1, so we mask out negatives in addition to {va, vb}.
    tri_a  = f2v[fa]
    tri_b  = f2v[fb]
    mask_a = (tri_a != va[:, None]) & (tri_a != vb[:, None]) & (tri_a >= 0)
    mask_b = (tri_b != va[:, None]) & (tri_b != vb[:, None]) & (tri_b >= 0)
    oa     = tri_a[mask_a].reshape(-1)
    ob     = tri_b[mask_b].reshape(-1)

    # Quad winding: va -> oa -> vb -> ob.  By construction the removed edge
    # (va, vb) is one of the quad's two diagonals.
    quad = np.stack([va, oa, vb, ob], axis=1)
    p    = points[quad]  # (N, 4, 2 or 3)

    # 1. planarity (dihedral) - 1.0 for 2D since 2D is always coplanar
    if is_2d:
        planarity = np.ones(cand.size, dtype=float)
    else:
        planarity = np.einsum("ij,ij->i", face_normals[fa], face_normals[fb])
        planarity = (planarity + 1.0) * 0.5

    # 2. squareness - 1 when all four corners are 90 deg, 0 when degenerate
    e0  = _l2_normalize(p[:, 1] - p[:, 0])
    e1  = _l2_normalize(p[:, 2] - p[:, 1])
    e2_ = _l2_normalize(p[:, 3] - p[:, 2])
    e3  = _l2_normalize(p[:, 0] - p[:, 3])
    cos_corners = np.stack(
        [
            np.einsum("ij,ij->i", -e3, e0),
            np.einsum("ij,ij->i", -e0, e1),
            np.einsum("ij,ij->i", -e1, e2_),
            np.einsum("ij,ij->i", -e2_, e3),
        ],
        axis=1,
    )
    squareness = 1.0 - np.abs(cos_corners).mean(axis=1)

    # 3. diagonal-dominance and balance
    L_rm  = np.linalg.norm(points[vb] - points[va], axis=1)
    L_alt = np.linalg.norm(points[ob] - points[oa], axis=1)
    L_side = np.stack(
        [np.linalg.norm(p[:, (i + 1) % 4] - p[:, i], axis=1) for i in range(4)],
        axis=1,
    ).mean(axis=1)
    diag_dom     = np.tanh(L_rm / (L_side + EPSILON) - 1.0) * 0.5 + 0.5
    diag_balance = np.minimum(L_rm, L_alt) / (np.maximum(L_rm, L_alt) + EPSILON)
    diag_score   = diag_dom * diag_balance

    # 4. convexity - in 3D, sign of cross product . averaged face normal
    #               in 2D, sign of the scalar 2D cross product
    if is_2d:
        signs = np.stack(
            [
                _cross_2d(
                    p[:, (i + 1) % 4] - p[:, i],
                    p[:, (i + 2) % 4] - p[:, (i + 1) % 4],
                )
                for i in range(4)
            ],
            axis=1,
        )
    else:
        n_avg = _l2_normalize(face_normals[fa] + face_normals[fb])
        signs = np.stack(
            [
                np.einsum(
                    "ij,ij->i",
                    np.cross(
                        p[:, (i + 1) % 4] - p[:, i],
                        p[:, (i + 2) % 4] - p[:, (i + 1) % 4],
                    ),
                    n_avg,
                )
                for i in range(4)
            ],
            axis=1,
        )
    convex = (signs > 0).all(axis=1) | (signs < 0).all(axis=1)

    score = (
        planarity_weight * planarity
        + squareness_weight * squareness
        + diagonal_weight * diag_score
    )
    if require_convex:
        score = np.where(convex, score, -np.inf)
    score = np.where(score >= min_score, score, -np.inf)

    return score, fa, fb, cand