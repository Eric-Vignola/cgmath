from dataclasses import dataclass

import numpy as np
from cgmath.data import Array, Data
from cgmath.geometry.utils import (
    bezier_raycast,
    bilinear_integrate,
    bilinear_raycast,
    bilinear_vectors,
    compute_centroids,
    compute_samples,
    remap_indices,
)
from cgmath.geometry.utils.main import _seeded_bilinear_sample
from scipy.spatial import cKDTree

# --------------------- SADDLE SURFACE SAMPLE ALGORITHM ---------------------- #


def _quad_geometry(geometry) -> np.ndarray:
    """face vertex indices as int32 quads: triangles padded with -1"""
    geometry = np.asarray(geometry, dtype=np.int32)
    if geometry.shape[1] > 4:
        raise ValueError("ngons detected")

    elif geometry.shape[1] < 4:
        buffer                         = np.full((geometry.shape[0], 4), -1, dtype=np.int32)
        buffer[:, : geometry.shape[1]] = geometry
        geometry                       = buffer

    return geometry


def integrate(points, geometry, samples=10, method="gauss", tolerance=1e-4):
    """
    integrates the given topology and returns the surface area approximation
    """

    # make sure geometry is shaped for quads
    geometry = _quad_geometry(geometry)

    # make sure points are 3D
    if points.shape[1] > 3:
        points = points[:, :3]

    elif points.shape[1] < 3:
        buffer                       = np.zeros((points.shape[0], 3), dtype=points.dtype)
        buffer[:, : points.shape[1]] = points
        points                       = buffer

    # approximate the saddle surface area
    area = bilinear_integrate(
        points    = points,
        geometry  = geometry,
        samples   = samples,
        method    = method,
        tolerance = tolerance,
    )
    return np.round(area, decimals=16)


@dataclass(repr=False, eq=False)
class SampleData(Data):
    """
    A dataclass to hold saddle surface sample data
    """

    projections: Array(np.float64, "N", "D")  # closest projection
    distances:   Array(np.float64, "N")       # distance to projection
    weights:     Array(np.float64, "N", 4)    # influence weight of face points
    indices:     Array(np.int32, "N")         # closest face indices
    normals:     Array(np.float64, "N", "D")  # closest face normals
    occluded:    Array(np.bool_, "N")         # are the queried points contained by the mesh
    uvs:         Array(np.float64, "N", 2)    # closest saddle surface uv value
    geometry:    Array(np.int32, "N", 4)      # face vertices, -1 padded for triangles

    def __call__(self, values):
        return self.compute(values)

    def compute(self, values):
        """computes weighted values from given data"""

        return compute_samples(values, self.weights, self.geometry)

    def remap(self, src_mesh, dst_mesh, dst_uv):
        """
        saddle surface sampler where closest face index is returned
        as a geometric vertex index relationship
        """
        sample_data = self.copy()

        indices = remap_indices(dst_mesh.indices, dst_uv.indices, self.distances)

        sample_data.projections = sample_data.projections[indices]
        sample_data.distances   = sample_data.distances[indices]
        sample_data.weights     = sample_data.weights[indices]
        sample_data.indices     = sample_data.indices[indices]
        sample_data.normals     = sample_data.normals[indices]
        sample_data.occluded    = sample_data.occluded[indices]
        sample_data.uvs         = sample_data.uvs[indices]
        sample_data.geometry    = _quad_geometry(src_mesh.geometry)[sample_data.indices]

        return sample_data


@dataclass(repr=False, eq=False)
class RaycastData(Data):
    """
    A dataclass to hold ray-mesh intersection data
    """

    projections: Array(np.float64, "N", "D")  # intersection points (origins for misses)
    distances:   Array(np.float64, "N")       # ray t values (distance along ray to intersection)
    weights:     Array(np.float64, "N", 4)    # bilinear weights of the 4 face vertices at hit point
    indices:     Array(np.int32, "N")         # face indices of hits (-1 for misses)
    normals:     Array(np.float64, "N", "D")  # interpolated face normals at hit points
    occluded:    Array(np.bool_, "N")         # boolean mask for occlusion
    uvs:         Array(np.float64, "N", 2)    # bilinear (u, v) parameters at hit points
    geometry:    Array(np.int32, "N", 4)      # face vertex indices for hit faces
    hit:         Array(np.bool_, "N")         # boolean mask -- True where a face was intersected

    def __call__(self, values):
        return self.compute(values)

    def compute(self, values):
        """computes weighted values from given data"""
        return compute_samples(values, self.weights, self.geometry)


def raycast(
    points,
    face_points,
    face_geometry,
    origins,
    directions,
    normals         = None,
    surface_normals = None,
    forward_only    = True,
    twosided        = True,
    bvh             = None,
):
    """
    Ray-mesh intersection on bilinear or bicubic Bezier patches.

    When surface_normals are provided, constructs PN Quad bicubic Bezier
    patches for higher accuracy on smooth meshes. Otherwise falls back to
    bilinear patches.

    Casts rays against a mesh and returns intersection data. Directions
    need not be unit length: a distance is measured along the unit ray,
    and a projection is the point hit.
    If forward_only=False, also casts backward and keeps the closer hit:
    a backward hit's distance is its length along -direction, positive
    like a forward one, and its projection is the point hit. Its occluded
    flag stays relative to the direction given, so a ray and its reverse
    report opposite flags for the one hit.
    When twosided=False, back-face hits are rejected at the Numba level.

    When ``bvh`` is supplied (a pre-built ``BVHData`` from
    ``cgmath.geometry.utils._numba._bvh.build_bvh``), the BVH-accelerated
    raycast kernel is used.  The BVH is reused for both forward and
    backward casts when ``forward_only=False``.
    """
    # pad geometry to 4-wide if needed
    face_geometry = _quad_geometry(face_geometry)

    n = origins.shape[0]

    _cast = bezier_raycast if surface_normals is not None else bilinear_raycast

    def _do_cast(orig, dirs):
        if surface_normals is not None:
            return _cast(
                face_points,
                face_geometry,
                surface_normals,
                orig,
                dirs,
                twosided = twosided,
                bvh      = bvh,
            )
        return _cast(
            face_points,
            face_geometry,
            orig,
            dirs,
            twosided = twosided,
            bvh      = bvh,
        )

    # forward cast
    t, uv, face_indices = _do_cast(origins, directions)

    # backward cast if requested
    if not forward_only:
        t_bwd, uv_bwd, face_bwd = _do_cast(origins, -directions)
        # keep the closer hit (compare absolute t values)
        closer               = (~np.isnan(t_bwd)) & (np.isnan(t) | (t_bwd < t))
        t[closer]            = t_bwd[closer]
        uv[closer]           = uv_bwd[closer]
        face_indices[closer] = face_bwd[closer]
    else:
        closer = np.zeros(n, dtype=bool)

    # compute bilinear weights from UV
    u             = uv[:, 0]
    v             = uv[:, 1]
    ou            = 1.0 - u
    ov            = 1.0 - v
    weights       = np.empty((n, 4), dtype=np.float64)
    weights[:, 0] = ou * ov
    weights[:, 1] = u * ov
    weights[:, 2] = u * v
    weights[:, 3] = ou * v

    # triangle adjustment: fold w3 into w2 when face vertex 3 is -1
    hit_mask = face_indices >= 0
    for i in range(n):
        if hit_mask[i] and face_geometry[face_indices[i], 3] == -1:
            weights[i, 2] += weights[i, 3]
            weights[i, 3] = 0.0

    # compute projections: the kernels cast unit rays, so t is a distance
    # along the unit direction, and a backward hit's runs along -direction.
    # A zero direction misses: its NaN is reset to the origin below
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    projections = origins + np.where(closer, -t, t)[:, None] * unit

    # gather hit face geometry
    safe_indices            = np.where(hit_mask, face_indices, 0)
    hit_geometry            = face_geometry[safe_indices]
    hit_geometry[~hit_mask] = -1

    # compute normals and occluded flag
    if normals is not None:
        result                   = normals[hit_geometry] * weights[:, :, None]
        result[hit_geometry < 0] = 0.0
        interp_normals           = np.sum(result, axis=1)
        mag                      = np.einsum("...i,...i", interp_normals, interp_normals) ** 0.5
        mag[mag == 0]            = 1.0
        interp_normals /= mag[:, None]
        occluded = np.einsum("...i,...i", directions, interp_normals) < 0
    else:
        interp_normals = np.zeros_like(origins)
        occluded       = np.full(n, False)

    # hit mask
    hit = face_indices >= 0

    # for misses, set projections to the original ray origins
    projections[~hit] = origins[~hit]

    # distances: set misses to nan
    distances = t.copy()

    return RaycastData(
        projections = projections,
        distances   = distances,
        weights     = weights,
        indices     = face_indices,
        normals     = interp_normals,
        occluded    = occluded,
        uvs         = uv,
        geometry    = hit_geometry,
        hit         = hit,
    )


def sample(
    points,
    face_points,
    face_geometry,
    normals                 = None,
    surface_normals         = None,
    iteration_count         = 100,
    iteration_tolerance     = 1e-8,
    uv_border_tolerance     = 1e-5,
    perfect_match_tolerance = 1e-6,
):
    """
    preprocess data before going into the saddle surface algorithm
    1- weed out exact matches is a cKDTree (much faster!): a query closer to
       a vertex than perfect_match_tolerance times the shortest edge of the
       faces around it gets that vertex, at its real distance. A vertex no
       face holds is never matched, wherever it is stored, so it changes no
       answer
    2- if not 100% match, precompute face centroids and radiuses for optimization.
       and send what remains to saddle surface sampler
    """
    from cgmath.geometry.utils._numba._topology import _vertex_faces

    # the results are float64 whatever the queries are: int queries would
    # truncate every weight to 0
    points      = np.asarray(points)
    projections = np.array(points, dtype=np.float64)
    distances   = np.zeros(points.shape[0], dtype=np.float64)
    weights     = np.empty((points.shape[0], 4), dtype=np.float64)
    indices     = np.empty(points.shape[0],      dtype=np.int32)
    uvs         = np.empty((points.shape[0], 2), dtype=np.float64)

    # shaped for quads before anything reads it: the result's geometry is
    # 4 wide even when every query matches a vertex
    face_geometry = _quad_geometry(face_geometry)

    # sampling happens only on min dimension
    min_width = min(points.shape[1], face_points.shape[1])

    # per vertex, in one pass over the faces: the first face that holds it
    # and its slot there, -1 for a vertex no face holds, and the shortest
    # edge of all the faces that hold it
    vertex_faces, vertex_slots, shortest = _vertex_faces(
        np.asarray(face_points[:, :min_width], dtype=np.float64), face_geometry
    )

    # do a quick kdtree lookup to weed out perfect matches. Only the vertices
    # a face holds are looked up: a vertex no face holds has no face to
    # answer with, and a copy of a vertex, stored before it, would take its
    # queries away
    held = vertex_faces >= 0
    if np.all(held):
        tree = cKDTree(face_points[:, :min_width])
        dist, idx = tree.query(points[:, :min_width])

    else:
        held_vertices = np.nonzero(held)[0]
        tree          = cKDTree(face_points[held_vertices, :min_width])
        dist, idx = tree.query(points[:, :min_width])
        idx = held_vertices[idx]

    # the match radius is a fraction of the shortest edge around the vertex,
    # so it means the same on a mesh of any size, and in any face order
    matched = dist <= perfect_match_tolerance * shortest[idx]

    # if any matches, set their weights to 1.0
    if np.any(matched):
        f_ = vertex_faces[idx[matched]]
        i_ = vertex_slots[idx[matched]]
        w  = np.zeros((i_.size, 4))
        w[np.arange(i_.size), i_] = 1
        weights[matched]          = w
        indices[matched]          = f_
        uvs[matched, 0]           = w[:, 1] + w[:, 2]
        uvs[matched, 1]           = w[:, 2] + w[:, 3]

        # the vertex itself, at its real distance from the query
        projections[matched, :min_width] = face_points[idx[matched], :min_width]
        distances[matched]               = dist[matched]

    # use saddle surface sampler for non 100% matches
    if not np.all(matched):
        unmatched = ~matched
        centroids, radiuses = compute_centroids(
            face_points[:, :min_width], face_geometry, return_radiuses=True
        )

        kdtree   = cKDTree(centroids)
        max_dist = kdtree.query(points[unmatched, :min_width])[0].max() ** 2

        # a face of each query's nearest vertex, which the sampler solves
        # first to skip the faces beyond it
        proj, d, w, uv, i = _seeded_bilinear_sample(
            points[unmatched, :min_width],
            face_points[:, :min_width],
            face_geometry,
            centroids,
            radiuses,
            centroid_distance_tolerance=max_dist,
            seed_faces=vertex_faces[idx[unmatched]],
            iteration_count=iteration_count,
            iteration_tolerance=iteration_tolerance,
            uv_border_tolerance=uv_border_tolerance,
            normals=surface_normals,
        )

        projections[unmatched, :min_width] = proj
        distances[unmatched]               = d**0.5
        weights[unmatched]                 = w
        indices[unmatched]                 = i
        uvs[unmatched]                     = uv

    # calculate the contained vertices with the given normals
    if normals is not None:
        result   = normals[face_geometry[indices]] * weights[:, :, None]
        normals  = np.sum(result, axis=1)
        v        = projections - points
        occluded = np.einsum("...i,...i", v, normals) > 0

    else:
        occluded = np.full(points.shape[0], False)
        normals  = np.zeros(points.shape, dtype=np.float64)

    return SampleData(
        projections = projections,
        distances   = distances,
        weights     = weights,
        indices     = indices,
        normals     = normals,
        occluded    = occluded,
        uvs         = uvs,
        geometry    = face_geometry[indices],
    )