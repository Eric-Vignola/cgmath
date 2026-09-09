import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True, cache=True)
def _classify_cubes(scalar_field, iso_value):
    """
    Classify all cubes based on corner values.
    Returns cube_config (8-bit) for each cube where each bit represents
    whether a corner is inside (>=iso_value) or outside (<iso_value).

    Corner ordering (standard marching cubes):
        4----5
       /|   /|
      7----6 |
      | 0--|-1
      |/   |/
      3----2

    Bit positions: corner 0 = bit 0, corner 1 = bit 1, etc.
    """
    nx, ny, nz = scalar_field.shape
    cube_config = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.uint8)

    for i in prange(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                config = 0

                # Corner 0: (i, j, k)
                if scalar_field[i, j, k] >= iso_value:
                    config |= 1
                # Corner 1: (i+1, j, k)
                if scalar_field[i + 1, j, k] >= iso_value:
                    config |= 2
                # Corner 2: (i+1, j+1, k)
                if scalar_field[i + 1, j + 1, k] >= iso_value:
                    config |= 4
                # Corner 3: (i, j+1, k)
                if scalar_field[i, j + 1, k] >= iso_value:
                    config |= 8
                # Corner 4: (i, j, k+1)
                if scalar_field[i, j, k + 1] >= iso_value:
                    config |= 16
                # Corner 5: (i+1, j, k+1)
                if scalar_field[i + 1, j, k + 1] >= iso_value:
                    config |= 32
                # Corner 6: (i+1, j+1, k+1)
                if scalar_field[i + 1, j + 1, k + 1] >= iso_value:
                    config |= 64
                # Corner 7: (i, j+1, k+1)
                if scalar_field[i, j + 1, k + 1] >= iso_value:
                    config |= 128

                cube_config[i, j, k] = config

    return cube_config


@njit(fastmath=True, parallel=True, cache=True)
def _count_active_cubes(cube_config):
    """
    Count cubes where isosurface passes through (0 < config < 255).
    Returns count per slice for efficient parallel extraction.
    """
    nx, ny, nz = cube_config.shape
    counts = np.zeros(nx, dtype=np.int64)

    for i in prange(nx):
        count = 0
        for j in range(ny):
            for k in range(nz):
                config = cube_config[i, j, k]
                if config != 0 and config != 255:
                    count += 1
        counts[i] = count

    return counts


@njit(fastmath=True, parallel=True, cache=True)
def _extract_active_cubes(cube_config, counts, total_count):
    """
    Extract active cube coordinates into a flat array.
    Uses pre-computed counts for correct indexing.
    """
    nx, ny, nz = cube_config.shape
    active_cubes = np.empty((total_count, 3), dtype=np.int32)

    # Compute offsets from counts
    offsets = np.zeros(nx + 1, dtype=np.int64)
    for i in range(nx):
        offsets[i + 1] = offsets[i] + counts[i]

    for i in prange(nx):
        idx = offsets[i]
        for j in range(ny):
            for k in range(nz):
                config = cube_config[i, j, k]
                if config != 0 and config != 255:
                    active_cubes[idx, 0] = i
                    active_cubes[idx, 1] = j
                    active_cubes[idx, 2] = k
                    idx += 1

    return active_cubes


@njit(fastmath=True, parallel=True, cache=True)
def _build_cube_to_vertex_map(active_cubes, nx, ny, nz):
    """
    Build a 3D lookup table mapping cube coordinates to vertex indices.
    Returns -1 for inactive cubes.
    """
    cube_to_vertex = np.full((nx, ny, nz), -1, dtype=np.int32)

    for idx in prange(active_cubes.shape[0]):
        i = active_cubes[idx, 0]
        j = active_cubes[idx, 1]
        k = active_cubes[idx, 2]
        cube_to_vertex[i, j, k] = idx

    return cube_to_vertex


@njit(fastmath=True, parallel=True, cache=True)
def _compute_dual_vertices(
    scalar_field,
    cube_config,
    active_cubes,
    iso_value,
    grid_origin,
    grid_spacing,
    edge_table,
    edge_vertices,
    corner_offsets,
):
    """
    Compute dual vertex positions for each active cube.
    Position = centroid of edge crossing points (linear interpolation).
    """
    n_active = active_cubes.shape[0]
    points   = np.empty((n_active, 3), dtype=np.float64)

    for idx in prange(n_active):
        i         = active_cubes[idx, 0]
        j         = active_cubes[idx, 1]
        k         = active_cubes[idx, 2]

        config    = cube_config[i, j, k]
        edge_bits = edge_table[config]

        # Accumulate edge crossing positions
        cx, cy, cz = 0.0, 0.0, 0.0
        count = 0

        for e in range(12):
            if edge_bits & (1 << e):
                # Get corner indices for this edge
                c0 = edge_vertices[e, 0]
                c1 = edge_vertices[e, 1]

                # Get corner offsets
                o0x, o0y, o0z = (
                    corner_offsets[c0, 0],
                    corner_offsets[c0, 1],
                    corner_offsets[c0, 2],
                )
                o1x, o1y, o1z = (
                    corner_offsets[c1, 0],
                    corner_offsets[c1, 1],
                    corner_offsets[c1, 2],
                )

                # Get scalar values at corners
                v0 = scalar_field[i + o0x, j + o0y, k + o0z]
                v1 = scalar_field[i + o1x, j + o1y, k + o1z]

                # Linear interpolation parameter
                dv = v1 - v0
                if abs(dv) < 1e-10:
                    t = 0.5
                else:
                    t = (iso_value - v0) / dv

                # Compute world position of crossing
                p0x = grid_origin[0] + (i + o0x) * grid_spacing[0]
                p0y = grid_origin[1] + (j + o0y) * grid_spacing[1]
                p0z = grid_origin[2] + (k + o0z) * grid_spacing[2]

                p1x = grid_origin[0] + (i + o1x) * grid_spacing[0]
                p1y = grid_origin[1] + (j + o1y) * grid_spacing[1]
                p1z = grid_origin[2] + (k + o1z) * grid_spacing[2]

                cx += p0x + t * (p1x - p0x)
                cy += p0y + t * (p1y - p0y)
                cz += p0z + t * (p1z - p0z)
                count += 1

        # Dual vertex is centroid of crossings
        if count > 0:
            inv_count = 1.0 / count
            points[idx, 0] = cx * inv_count
            points[idx, 1] = cy * inv_count
            points[idx, 2] = cz * inv_count
        else:
            # Fallback to cube center (should not happen for active cubes)
            points[idx, 0] = grid_origin[0] + (i + 0.5) * grid_spacing[0]
            points[idx, 1] = grid_origin[1] + (j + 0.5) * grid_spacing[1]
            points[idx, 2] = grid_origin[2] + (k + 0.5) * grid_spacing[2]

    return points


# Edges that this cube "owns" - edges emanating from corner 0 along +X, +Y, +Z
# Edge 0: corner 0 -> corner 1 (along +X)
# Edge 3: corner 0 -> corner 3 (along +Y)
# Edge 8: corner 0 -> corner 4 (along +Z)
# Other edges are owned by adjacent cubes, so we skip them to avoid duplicates.
OWNED_EDGES = (0, 3, 8)


@njit(fastmath=True, cache=True)
def _count_faces(
    active_cubes,
    cube_to_vertex,
    cube_config,
    edge_table,
    edge_axis,
    face_cube_offsets_x,
    face_cube_offsets_y,
    face_cube_offsets_z,
    nx,
    ny,
    nz,
):
    """
    Count the number of quad faces to generate.
    A face is created for each edge crossing where the current cube
    is the "primary" cube (owns the edge).

    Only edges 0, 3, 8 are owned by each cube (edges from corner 0).
    This prevents duplicate faces since each edge is shared by 4 cubes.
    """
    face_count = 0

    for idx in range(active_cubes.shape[0]):
        i         = active_cubes[idx, 0]
        j         = active_cubes[idx, 1]
        k         = active_cubes[idx, 2]

        config    = cube_config[i, j, k]
        edge_bits = edge_table[config]

        for e in OWNED_EDGES:
            if not (edge_bits & (1 << e)):
                continue

            axis = edge_axis[e]

            # Select appropriate offset table based on axis
            if axis == 0:
                offsets = face_cube_offsets_x
            elif axis == 1:
                offsets = face_cube_offsets_y
            else:
                offsets = face_cube_offsets_z

            # Check if this cube is the primary cube for this face
            # We use the cube with the LARGEST coordinates as primary.
            # Since our offsets are [0,0,0], [-1,0,0], etc., the current cube
            # (at offset [0,0,0]) has the largest coordinates.
            all_valid = True

            for n in range(4):
                ci = i + offsets[n, 0]
                cj = j + offsets[n, 1]
                ck = k + offsets[n, 2]

                # Check bounds
                if ci < 0 or ci >= nx or cj < 0 or cj >= ny or ck < 0 or ck >= nz:
                    all_valid = False
                    break

                # Check if cube is active
                if cube_to_vertex[ci, cj, ck] < 0:
                    all_valid = False
                    break

            if all_valid:
                face_count += 1

    return face_count


@njit(fastmath=True, parallel=True, cache=True)
def _generate_faces(
    active_cubes,
    cube_to_vertex,
    cube_config,
    scalar_field,
    edge_table,
    edge_vertices,
    edge_axis,
    face_cube_offsets_x,
    face_cube_offsets_y,
    face_cube_offsets_z,
    corner_offsets,
    nx,
    ny,
    nz,
):
    """
    Generate quad faces by connecting 4 cubes sharing each crossed edge.
    Returns face indices as a flat array and face count.

    Two-phase approach:
    1. First pass: count faces per active cube (for indexing)
    2. Second pass: generate face indices
    """
    n_active = active_cubes.shape[0]

    # Phase 1: Count faces per active cube
    # Only process owned edges (0, 3, 8) to avoid duplicate faces
    face_counts = np.zeros(n_active, dtype=np.int32)

    for idx in prange(n_active):
        i         = active_cubes[idx, 0]
        j         = active_cubes[idx, 1]
        k         = active_cubes[idx, 2]

        config    = cube_config[i, j, k]
        edge_bits = edge_table[config]
        count     = 0

        for e in OWNED_EDGES:
            if not (edge_bits & (1 << e)):
                continue

            axis = edge_axis[e]

            if axis == 0:
                offsets = face_cube_offsets_x
            elif axis == 1:
                offsets = face_cube_offsets_y
            else:
                offsets = face_cube_offsets_z

            all_valid = True

            for n in range(4):
                ci = i + offsets[n, 0]
                cj = j + offsets[n, 1]
                ck = k + offsets[n, 2]

                if ci < 0 or ci >= nx or cj < 0 or cj >= ny or ck < 0 or ck >= nz:
                    all_valid = False
                    break

                if cube_to_vertex[ci, cj, ck] < 0:
                    all_valid = False
                    break

            if all_valid:
                count += 1

        face_counts[idx] = count

    # Compute total faces and offsets
    total_faces = 0
    for idx in range(n_active):
        total_faces += face_counts[idx]

    face_offsets = np.zeros(n_active + 1, dtype=np.int64)
    for idx in range(n_active):
        face_offsets[idx + 1] = face_offsets[idx] + face_counts[idx]

    # Phase 2: Generate face indices
    # Only process owned edges (0, 3, 8) to avoid duplicate faces
    faces = np.empty((total_faces, 4), dtype=np.int32)

    for idx in prange(n_active):
        i         = active_cubes[idx, 0]
        j         = active_cubes[idx, 1]
        k         = active_cubes[idx, 2]

        config    = cube_config[i, j, k]
        edge_bits = edge_table[config]
        face_idx  = face_offsets[idx]

        for e in OWNED_EDGES:
            if not (edge_bits & (1 << e)):
                continue

            axis = edge_axis[e]

            if axis == 0:
                offsets = face_cube_offsets_x
            elif axis == 1:
                offsets = face_cube_offsets_y
            else:
                offsets = face_cube_offsets_z

            all_valid      = True
            vertex_indices = np.empty(4, dtype=np.int32)

            for n in range(4):
                ci = i + offsets[n, 0]
                cj = j + offsets[n, 1]
                ck = k + offsets[n, 2]

                if ci < 0 or ci >= nx or cj < 0 or cj >= ny or ck < 0 or ck >= nz:
                    all_valid = False
                    break

                vi = cube_to_vertex[ci, cj, ck]
                if vi < 0:
                    all_valid = False
                    break

                vertex_indices[n] = vi

            if all_valid:
                # Determine winding order based on gradient direction
                c0 = edge_vertices[e, 0]
                c1 = edge_vertices[e, 1]

                # Get corner offsets for gradient check
                o0x = corner_offsets[c0, 0]
                o0y = corner_offsets[c0, 1]
                o0z = corner_offsets[c0, 2]

                o1x = corner_offsets[c1, 0]
                o1y = corner_offsets[c1, 1]
                o1z = corner_offsets[c1, 2]

                v0  = scalar_field[i + o0x, j + o0y, k + o0z]
                v1  = scalar_field[i + o1x, j + o1y, k + o1z]

                # Reverse winding if gradient is negative
                if v0 < v1:
                    faces[face_idx, 0] = vertex_indices[0]
                    faces[face_idx, 1] = vertex_indices[1]
                    faces[face_idx, 2] = vertex_indices[2]
                    faces[face_idx, 3] = vertex_indices[3]
                else:
                    faces[face_idx, 0] = vertex_indices[3]
                    faces[face_idx, 1] = vertex_indices[2]
                    faces[face_idx, 2] = vertex_indices[1]
                    faces[face_idx, 3] = vertex_indices[0]

                face_idx += 1

    return faces, total_faces