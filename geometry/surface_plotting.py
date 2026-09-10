# TODO Once we iron out retargeting process we should refactor this to data classes
# also this is currently heavily in a prototyping stage and will eventually port to DDAP nodes

import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from cgmath.geometry.bspline import BSplineData
from transforms import (
    matrix_delta,
    matrix_multiply,
    matrix_weighted_transformation,
    vector_cross,
    vector_magnitude,
    vector_normalize,
    vector_to_matrix,
)


def surface_points_to_transformation_matrices(
    positions: np.ndarray, u_vecs: np.ndarray, v_vecs: np.ndarray
) -> dict:
    """
    Converts surface points and their corresponding U and V vectors into transformation matrices.

    Parameters:
    positions (numpy.ndarray): An array of 3D positions representing points on a surface.
    u_vecs (numpy.ndarray): An array of vectors representing the U direction on the surface.
    v_vecs (numpy.ndarray): An array of vectors representing the V direction on the surface.

    Returns:
    dict: A dictionary containing the transformation matrices, final positions, normalized U and V vectors, and normals.
    """
    u_vecs  = vector_normalize(u_vecs)
    v_vecs  = vector_normalize(v_vecs)

    normals = vector_cross(u_vecs, v_vecs)
    normals = vector_normalize(normals)
    # assumes that world up is Y/V and Aim vector is Z/N
    matrices = vector_to_matrix(normals, v_vecs, aim_axis=2, up_axis=1)

    final_positions    = np.copy(positions)
    matrices[:, 3, :3] = final_positions

    return {
        "matrices": matrices,
        "points":   final_positions,
        "u_vecs":   u_vecs,
        "v_vecs":   v_vecs,
        "normals":  normals,
    }


def convert_points_to_surfacespace_matrices(
    input_points: list, Mesh: object, UVList: list, uv_map_index: int = 0
) -> np.ndarray:
    """
    Converts input points to transformation matrices in surface space.

    Parameters:
    input_points (list): A list of points to be converted.
    Mesh (object): The mesh object containing geometry and points data.
    UVList (list): A list of UV data objects for sampling.
    uv_map_index (int): The index of the UV set to use. Default is 0.

    Returns:
    numpy.ndarray: An array of transformation matrices in surface space.
    """
    uvpoints = get_points_from_surface(
        Mesh, UVList, input_points, uv_map_index=uv_map_index, as_3d=True
    )
    return control_array_type_conversion(uvpoints, to_transformation_matrix=True)


def get_points_from_surface(
    Mesh:         object,
    UVList:       list,
    input_points: list,
    uv_map_index: int    = 0,
    as_3d:        bool   = False,
) -> np.ndarray:
    """
    Retrieves points from a surface based on input points and UV mapping.

    Parameters:
    Mesh (object): The mesh object containing geometry and points data.
    UVList (list): A list of UV data objects for sampling.
    input_points (list): A list of points to be converted.
    uv_map_index (int): The index of the UV set to use. Default is 0.
    as_3d (bool): If True, returns points in 3D space with an additional dimension. Default is False.

    Returns:
    numpy.ndarray: An array of points in either 2D or 3D space, depending on the as_3d parameter.
    """
    # if point arrays are two dimenisional it assumes uvs
    # and will return points in cartesian space
    if len(input_points[0]) == 2:
        sample_data = UVList[uv_map_index].sample(input_points)
        # convert uvfaces to geo faces.
        geo_face_indices = Mesh.geometry[sample_data.indices]
        geo_face_points = [
            Mesh.points[point_indices] for point_indices in geo_face_indices
        ]
    else:
        sample_data      = Mesh.sample(input_points)
        geo_face_indices = UVList[uv_map_index].geometry[sample_data.indices]
        geo_face_points = [
            UVList[uv_map_index].points[point_indices]
            for point_indices in geo_face_indices
        ]

    # final_positions = np.dot(sample_data.weights, geo_face_points)
    final_positions = np.array(
        [
            np.dot(weights, f_points)
            for weights, f_points in zip(sample_data.weights, geo_face_points)
        ]
    )

    if as_3d:
        final_positions = np.array([np.pad(arr, (0, 1)) for arr in final_positions])
    return final_positions


def control_array_type_conversion(
    array: list or np.ndarray, to_transformation_matrix: bool = False
) -> np.ndarray:
    """
    Converts an array of points or matrices into a different format based on the specified flag.

    Parameters:
    array (list or numpy.ndarray): The input array, which can be a list of 3D points or a list of transformation matrices.
    to_transformation_matrix (bool): If True, converts the input array to transformation matrices. If False, extracts translation components.

    Returns:
    numpy.ndarray: The converted array, either as transformation matrices or as an array of 3D points.
    """
    return_array = None

    if len(array[0]) == 3:
        if to_transformation_matrix:
            # build an array of identity matrices
            return_array = np.tile(np.eye(4), (len(array), 1, 1))
            # replace the translation
            return_array[:, 3, :3] = np.array(array)
        else:
            return_array = np.array(array)
    else:
        if to_transformation_matrix:
            return_array = np.array(array)
        else:
            return_array = np.array([matrx[3, :3] for matrx in array])

    return return_array


def transforms_to_surface_space(
    Mesh:         object,
    UVList:       list,
    matrices:     np.ndarray,
    uv_map_index: int        = 0,
    as_matrices:  bool       = False,
) -> np.ndarray:
    """
    Converts transformation matrices to surface space coordinates.

    Parameters:
    Mesh (object): The mesh object containing geometry and points data.
    UVList (list): A list of UV data objects for sampling.
    matrices (numpy.ndarray): An array of transformation matrices to be converted.
    uv_map_index (int): The index of the UV set to use. Default is 0.
    as_matrices (bool): If True, returns the result as transformation matrices. Default is False.

    Returns:
    numpy.ndarray: An array of points or transformation matrices in surface space.
    """
    points = control_array_type_conversion(matrices, to_transformation_matrix=False)
    # get uvn data with n=0
    surface_points = get_points_from_surface(
        Mesh, UVList, points, uv_map_index=uv_map_index, as_3d=True
    )
    if as_matrices:
        surface_points = control_array_type_conversion(
            surface_points, to_transformation_matrix=True
        )

    return surface_points


def transforms_from_surface_coordinates(
    Mesh:         object,
    UVList:       list,
    points:       list,
    uv_map_index: int    = 0,
    z_scale:      float  = 1.0,
) -> dict:
    """
    Transforms points from surface coordinates to 3D space using the given mesh and UV list.

    Parameters:
    Mesh (object): The mesh object containing geometry and points data.
    UVList (list): A list of UV data objects for sampling.
    points (list): A list of points to be transformed.
    uv_map_index (int): The index of the UV set to use. Default is 0.
    z_scale (float): The scale factor for the Z coordinate. Default is 1.0.

    Returns:
    dict: A dictionary containing transformed surface data including matrices, positions, u_vecs, v_vecs, and normals.
    """

    input_points = np.array(points)  # TODO deal with the the Z coordinate as a N offset
    uv_points    = input_points[:, :2]

    off      = 0.01
    u_off    = np.array([off, 0.0])
    v_off    = np.array([0.0, off])
    u_offset = np.array([np.add(uv, u_off) for uv in uv_points])
    v_offset = np.array([np.add(uv, v_off) for uv in uv_points])

    final_positions = get_points_from_surface(
        Mesh, UVList, uv_points, uv_map_index=uv_map_index
    )
    u_positions = get_points_from_surface(
        Mesh, UVList, u_offset, uv_map_index=uv_map_index
    )
    v_positions = get_points_from_surface(
        Mesh, UVList, v_offset, uv_map_index=uv_map_index
    )

    u_vecs = None
    v_vecs = None
    u_vecs = np.subtract(u_positions, final_positions)
    v_vecs = np.subtract(v_positions, final_positions)

    surface_data = surface_points_to_transformation_matrices(
        final_positions, u_vecs, v_vecs
    )

    return surface_data


def u_vector_to_rotation_matrix(input_vector: np.ndarray) -> np.ndarray:
    """
    Converts a given U vector into a rotation matrix.

    Parameters:
    input_vector (numpy.ndarray): A 3D vector representing the U direction.

    Returns:
    numpy.ndarray: A 4x4 transformation matrix with the U vector aligned to the X-axis.
    """
    # print(f"\ninput vec = {input_vector}")
    u_vec = np.copy(input_vector)
    u_vec[1] *= -1
    u_tangent = vector_normalize(np.array(u_vec))[0]
    normal    = np.array([0.0, 0.0, 1.0])

    # Compute the binormal vector (not needed for this case)
    # binormal = normal ^ u_tangent
    binormal = vector_normalize(vector_cross(normal, u_tangent))[0]
    # we compute a binormal to make sure y is perpendicular
    transform_matrix = np.array(
        [
            u_tangent[0],
            binormal[0],
            normal[0],
            0,
            u_tangent[1],
            binormal[1],
            normal[1],
            0,
            u_tangent[2],
            binormal[2],
            normal[2],
            0,
            0.0,
            0.0,
            0.0,
            1,
        ]
    )
    return transform_matrix


def reset_scale(transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Resets the scale component of a transformation matrix to identity.
    Args:
        transformation_matrix (numpy.ndarray): A 4x4 transformation matrix.
    Returns:
        numpy.ndarray: The input transformation matrix with its scale component reset to identity.
    """
    # Decompose the transformation matrix into its components (using QR decomposition)
    # We assume that the input matrix is an affine transformation matrix (i.e., last row is [0, 0, 0, 1])
    # Extract the 3x3 sub-matrix representing rotation and scale
    trans_mat      = np.copy(transformation_matrix)
    rotation_scale = trans_mat[:3, :3]
    # Compute the scaling factors from the lengths of the column vectors
    scale_factors = np.linalg.norm(rotation_scale, axis=0)
    # Normalize the column vectors to obtain the rotation matrix
    rotation = rotation_scale / scale_factors
    # Reconstruct the transformation matrix with the scale component reset to identity
    new_transformation_matrix         = np.eye(4)  # Initialize with identity
    new_transformation_matrix[:3, :3] = rotation   # Set the rotation component
    new_transformation_matrix[3, :3] = trans_mat[
        3, :3
    ]  # Copy the translation component

    return new_transformation_matrix


def extract_scale_matrix(transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Extracts the scale components from a 4x4 transformation matrix.

    Parameters:
    transformation_matrix (numpy.ndarray): A 4x4 matrix from which to extract scale components.

    Returns:
    numpy.ndarray: A 4x4 matrix containing only the scale components.
    """
    # Ensure the input is a 4x4 matrix
    if transformation_matrix.shape != (4, 4):
        raise ValueError("The transformation matrix must be a 4x4 matrix.")

    # Extract the scale factors from the diagonal of the transformation matrix
    scale_x = np.linalg.norm(transformation_matrix[0, :3])
    scale_y = np.linalg.norm(transformation_matrix[1, :3])
    scale_z = np.linalg.norm(transformation_matrix[2, :3])

    # Create a new transformation matrix with just the scale components
    scale_matrix       = np.eye(4)
    scale_matrix[0, 0] = scale_x
    scale_matrix[1, 1] = scale_y
    scale_matrix[2, 2] = scale_z

    return scale_matrix


def set_value_by_path(dictionary: dict, path: list, value) -> None:
    """
    Set a value in a dictionary by a given path.
    Args:
        dictionary (dict): The dictionary to modify.
        path (list): A list of keys representing the path to the value.
        value: The new value to set.
    """
    current_dict = dictionary
    for key in path[:-1]:
        if key not in current_dict or not isinstance(current_dict[key], dict):
            raise ValueError(f"Invalid path: {path}")
        current_dict = current_dict[key]
    current_dict[path[-1]] = value


def find_key(dictionary: dict, key: str) -> list:
    """
    Find all instances of a specific key in a dictionary.
    Args:
        dictionary (dict): The dictionary to search.
        key (str): The key to find.
    Returns:
        list: A list of tuples containing the path to each instance of the key.
    """

    def _find_key(dictionary, key, path=None):
        results = []
        if not path:
            path = []
        for k, v in dictionary.items():
            new_path = path + [k]
            if k == key:
                results.append((new_path, v))
            elif isinstance(v, dict):
                results.extend(_find_key(v, key, new_path))
        return results

    return _find_key(dictionary, key)


def refactor_control_object_matrices(
    control_matrices: np.ndarray, refactor_data: dict, refactor_alignment: bool = False
) -> dict:
    """
    Refactors control object matrices based on provided refactor data.

    Parameters:
    control_matrices (numpy.ndarray): An array of control matrices to be refactored.
    refactor_data (dict): A dictionary containing refactor parameters such as origin, scale, vector, refactor_matrix, and refactored_matrices.
    refactor_alignment (bool): A flag indicating whether to apply refactor alignment. Default is False.

    Returns:
    dict: A dictionary containing the refactored world matrices and offset matrices.
    """
    matrices = np.copy(control_matrices)
    origin   = refactor_data["origin"]
    # scale = refactor_data["scale"]
    # vector = refactor_data["vector"]
    refactor_matrix = refactor_data["refactor_matrix"]
    init_matrices   = refactor_data["refactored_matrices"]

    offset_matrices = None

    if isinstance(refactor_data["refactor_matrix"], type(None)) or np.all(
        refactor_data["refactor_matrix"] == np.eye(4)
    ):
        return control_matrices

    # first remove the scale component of matrices
    scale_matrices = np.array([extract_scale_matrix(mat) for mat in matrices])

    # establish relative offsets from pivot
    pivot_matrix        = np.eye(4)
    pivot_matrix[3, :3] = origin

    # put the matrices relative offsets to the pivot
    offset_matrices = np.array([matrix_delta(pivot_matrix, mat) for mat in matrices])

    matrices = np.array(
        [matrix_multiply(mat, refactor_matrix)[0] for mat in offset_matrices]
    )

    # reset the scale component
    matrices = np.array([reset_scale(mat) for mat in matrices])

    # add back in the original scale
    matrices = matrix_multiply(scale_matrices, matrices)

    # now compute the offset matrices
    if not isinstance(init_matrices, type(None)):
        offset_matrices = matrix_delta(init_matrices, matrices)

    return {"world": matrices, "offset": offset_matrices}


def refactor_control_object_matrices_init(
    control_matrices:   np.ndarray,
    uvn_scale:          list       = None,
    refactor_inputs:    list       = None,
    refactor_alignment: bool       = False,
) -> dict:
    """
    Initializes the refactoring of control object matrices based on provided inputs.

    Parameters:
    control_matrices (numpy.ndarray): An array of control matrices to be refactored.
    uvn_scale (list): A list of scale factors for U, V, and N directions. Default is [1.0, 1.0, 1.0].
    refactor_inputs (list): A list of inputs for refactoring, including points and matrices.
    refactor_alignment (bool): A flag indicating whether to apply refactor alignment. Default is False.

    Returns:
    dict: A dictionary containing refactored matrices, translation, vector, scale, origin, and refactor matrix.
    """
    if not uvn_scale:
        uvn_scale = [1.0, 1.0, 1.0]

    refactor_data = {
        "refactored_matrices": None,
        "translate": [0.0, 0.0, 0.0],
        "vector": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "refactor_matrix": None,
    }

    matrices  = np.copy(control_matrices)
    uvn_scale = np.array(uvn_scale)
    control_points = control_array_type_conversion(
        matrices, to_transformation_matrix=False
    )
    control_start           = control_points[0]
    control_end             = control_points[-1]
    control_vector          = np.subtract(control_end, control_start)
    control_mag             = vector_magnitude(control_vector)
    refactor_data["origin"] = control_start

    transform_matrix = np.eye(4)
    # offset_matrices = []
    # return_matrices = []

    if not isinstance(refactor_inputs, type(None)):
        points = control_array_type_conversion(
            refactor_inputs, to_transformation_matrix=False
        )
        refactor_start  = np.array(points[0])
        refactor_end    = np.array(points[-1])
        refactor_vector = np.subtract(refactor_end, refactor_start)
        refactor_mag    = vector_magnitude(refactor_vector)
        if abs(refactor_mag) < 0.0000001:
            print(
                "----------------------refactor_control_object_matrices_init:  refactor inputs have no magnitude"
            )
            return
        scale_factor = refactor_mag / control_mag

        scale = scale_factor * np.array(uvn_scale)

        refactor_data["translate"] = refactor_start
        refactor_data["vector"]    = refactor_vector
        refactor_data["scale"]     = scale

        scale_matrix = np.array(
            [
                [scale[0], 0.0, 0.0, 0.0],
                [0.0, scale[1], 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        if refactor_alignment:
            vec_uv           = np.zeros(3)
            vec_uv[:2]       = refactor_vector[:2]
            transform_matrix = u_vector_to_rotation_matrix(vec_uv)

        transform_matrix        = matrix_multiply(scale_matrix, transform_matrix)[0]
        transform_matrix[3, :3] = refactor_start

        refactor_data["refactor_matrix"] = transform_matrix
        matrices = refactor_control_object_matrices(
            control_matrices, refactor_data, refactor_alignment=refactor_alignment
        )["world"]
        refactor_data["refactored_matrices"] = matrices
    else:
        print("do something")
        refactor_data["refactor_matrix"]     = np.eye(4)
        refactor_data["refactored_matrices"] = matrices

    return refactor_data


def get_bspline_map_init(
    control_data:       Tuple[List[str], np.ndarray],
    driven_data:        Optional[Tuple[List[str], np.ndarray]] = None,
    Mesh:               Optional[object]                       = None,
    UVList:             Optional[List[object]]                 = None,
    uv_map_index:       int                                    = 0,
    uvn_scale:          Optional[List[float]]                  = None,
    refactor_inputs:    Optional[Tuple[List[str], np.ndarray]] = None,
    refactor_alignment: bool                                   = False,
) -> Dict[str, Union[List, Dict]]:
    """
    Initializes a B-spline mapping for control and driven data, potentially refactoring the control matrices.

    Parameters:
    control_data (Tuple[List[str], np.ndarray]): A tuple containing a list of control names and their corresponding matrices.
    driven_data (Optional[Tuple[List[str], np.ndarray]]): A tuple containing a list of driven names and their corresponding matrices. Default is None.
    Mesh (Optional[object]): The mesh object for surface data. Default is None.
    UVList (Optional[List[object]]): A list of UV data objects for sampling. Default is None.
    uv_map_index (int): The index of the UV set to use. Default is 0.
    uvn_scale (Optional[List[float]]): A list of scale factors for U, V, and N directions. Default is [1.0, 1.0, 1.0].
    refactor_inputs (Optional[Tuple[List[str], np.ndarray]]): A tuple containing refactor input names and their corresponding matrices. Default is None.
    refactor_alignment (bool): A flag indicating whether to apply refactor alignment. Default is False.

    Returns:
    Dict[str, Union[List, Dict]]: A dictionary containing control names, parameters, basis, offset matrices, offset points, super sample plotting data, and refactor data.
    """
    if not uvn_scale:
        uvn_scale = [1.0, 1.0, 1.0]
    sample_num       = 200
    degree           = 3
    periodic         = False
    driven_points    = np.array([[0, 0, 0]])
    control_points   = []

    control_names    = control_data[0]
    control_matrices = control_data[1]

    driven_matrices  = None
    if driven_data:
        driven_names    = driven_data[0]
        driven_matrices = driven_data[1]

    rdata = {
        "controls": control_names,
        "params": [],
        "basis": [],
        "offset_matrices": [],
        "offset_points": [],
        "super_sample_plotting": [],
        "refactor_data": {
            "refactored_matrices": [],
            "translate": [0.0, 0.0, 0.0],
            "vector": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "origin": [0.0, 0.0, 0.0],
            "refactor_matrix": None,
        },
    }

    # if refactoring do it here
    if not isinstance(refactor_inputs, type(None)):
        points = control_array_type_conversion(refactor_inputs[1])
        # TODO here also make a version that does not require Mesh inputs
        surface_points = get_points_from_surface(
            Mesh, UVList, points, uv_map_index=uv_map_index
        )
        surface_points = np.array([np.pad(arr, (0, 1)) for arr in surface_points])
        # print(f"refactor_points (UV) = {surface_points}")
        refactor_matrices = control_array_type_conversion(
            surface_points, to_transformation_matrix=True
        )

        # print(refactor_inputs[0])
        refactor_data = refactor_control_object_matrices_init(
            control_matrices,
            uvn_scale          = uvn_scale,
            refactor_inputs    = refactor_matrices,
            refactor_alignment = refactor_alignment,
        )

        rdata["refactor_data"] = refactor_data
        control_matrices       = np.copy(refactor_data["refactored_matrices"])
        # if the input refactored matrices are greater than 2
        # we assume it is full bspline we are refactoring for
        # and we want to use it for mapping the driven objects.
        if len(refactor_matrices) > 2:
            control_matrices = refactor_matrices

    if len(control_matrices[0]) == 3:
        control_points = np.array(control_matrices)
    else:
        # control_points = np.array([matrx[3, :3] for matrx in control_matrices])
        control_points = control_matrices[:, 3, :3]

    if not isinstance(driven_matrices, type(None)):
        rdata["driven"] = driven_names
        # print(driven_matrices)
        driven_points = control_array_type_conversion(
            driven_matrices, to_transformation_matrix=False
        )

    # construct a bspline from the control_matrices
    spline = BSplineData(points=control_points, degree=degree, periodic=periodic)

    # now super sample to plot in surface space
    plotting_points = spline.compute(np.linspace(0, spline.max_param, sample_num))[0]

    param_curve = spline
    if Mesh:  # we need to convert surface coords to cartesian positions (bspline approximation of surface)
        surface_data = transforms_from_surface_coordinates(
            Mesh,
            UVList,
            plotting_points,
            uv_map_index = uv_map_index,
            z_scale      = uvn_scale[2],
        )

        rdata["super_sample_plotting"] = np.copy(surface_data["matrices"])
        # now use that super sample plotting to establish u coordinates of driven
        param_curve = BSplineData(
            points=surface_data["points"], degree=degree, periodic=periodic
        )
        # print(f"super sample points = {surface_data['points']}")
    else:
        rdata["super_sample_plotting"] = np.copy(plotting_points)

    # record the default positions of driven objects
    if not isinstance(
        driven_matrices, type(None)
    ):  # now get the actual parameters and offset data
        # get closest curve data
        driven_data = param_curve.sample(driven_points)

        # refactor params if super sampling
        param_factor    = spline.max_param / param_curve.max_param
        params          = driven_data.params * param_factor
        rdata["params"] = params
        # we want the Basis from the control spline not the super sampled one if super sampled
        plotting_points = spline.compute(params)[0]
        spline_data     = spline.sample(plotting_points)
        rdata["basis"]  = spline_data.basis

        if Mesh:  # we need to convert surface coords to cartesian positions (bspline approximation of surface)
            surface_data = transforms_from_surface_coordinates(
                Mesh,
                UVList,
                plotting_points,
                uv_map_index = uv_map_index,
                z_scale      = uvn_scale[2],
            )

            rdata["offset_matrices"] = matrix_delta(
                surface_data["matrices"], driven_matrices
            )
            rdata["offset_points"] = np.subtract(driven_points, surface_data["points"])
        else:  # this method is used if no mesh is involved.
            rdata["offset_points"] = np.subtract(driven_points, driven_data.points)
            offset_matrices        = np.copy(driven_matrices)
            for offset, point in zip(offset_matrices, rdata["offset_points"]):
                offset[3, :3] = point
            rdata["offset_matrices"] = offset_matrices

    return rdata


def build_normal_matrix(
    input_point: np.ndarray, normal_scale: float = 1.0
) -> np.ndarray:
    normal_offset        = np.zeros_like(input_point)
    normal_offset[2]     = input_point[2] * normal_scale
    normal_matrix        = np.eye(4)
    normal_matrix[3, :3] = normal_offset
    return normal_matrix


def bspline_weigh_transformations(
    control_data:    dict,
    bspline_map:     dict,
    Mesh:            Optional[object] = None,
    UVList:          Optional[list]   = None,
    uv_map_index:    int              = 0,
    normal_scale:    float            = 1.0,
    return_matrices: bool             = False,
) -> dict:
    """
    Computes the weighted transformation for objects based on B-spline mapping and optional surface constraints.

    Args:
        control_data (dict): Contains 'matrices' and 'points' for control transformations.
        bspline_map (dict): Mapping of objects to their B-spline basis and matrix offsets.
        Mesh (Mesh, optional): Surface Space to constrain transformations to. Defaults to None.
        normal_scale (float, optional): Scale for normal offset on the surface. Defaults to 1.0.
        return_matrices (bool, optional): If True, returns transformation matrices. Otherwise, returns positions. Defaults to False.

    Returns:
        dict: A dictionary mapping objects to their computed transformations or positions.
    """
    matrices = control_data["matrices"]
    points   = control_data["points"]
    rdata    = {"bspline_plot": [], "plotting": None, "transformation": []}

    # build the spline
    spline = BSplineData(points=points, degree=3, periodic=False)
    # max_param = spline.max_param

    # get the points
    plotting_points   = spline.compute(bspline_map["params"])[0]
    rdata["plotting"] = np.copy(plotting_points)
    offset_matrices   = np.copy(bspline_map["offset_matrices"])
    # get the surface data transformatin matrices
    if Mesh:
        surface_data = transforms_from_surface_coordinates(
            Mesh,
            UVList,
            plotting_points,
            uv_map_index = uv_map_index,
            z_scale      = normal_scale,
        )
        rdata["plotting"] = np.copy(surface_data["matrices"])

    for basis, offset_matrix, bspline_transform in zip(
        bspline_map["basis"], offset_matrices, rdata["plotting"]
    ):
        offset = np.copy(offset_matrix)
        # TODO migrate compute weighted transformation to matrix module
        set_matrix = matrix_weighted_transformation(
            matrices, weights=basis, flatten=False
        )

        # first we need to abstract the rotational and scale contribution
        bspl_point        = np.copy(set_matrix[3, :3])
        set_matrix[3, :3] = [0.0, 0.0, 0.0]
        normal_matrix = build_normal_matrix(
            input_point=bspl_point, normal_scale=normal_scale
        )

        if Mesh:
            surface_matrix = matrix_multiply(normal_matrix, bspline_transform)[0]
            set_matrix     = matrix_multiply(set_matrix, surface_matrix)[0]
            rdata["bspline_plot"].append(np.copy(set_matrix))
            set_matrix = matrix_multiply(offset_matrix, set_matrix)[0]
        else:
            offset[3, :3] += np.add(bspline_transform, offset[3, :3])
            set_matrix = matrix_multiply(set_matrix, offset)[0]

        if return_matrices:
            rdata["transformation"].append(set_matrix)
        else:
            rdata["transformation"].append(set_matrix[3, :3])

    return rdata


def control_sort(lst: List[str]) -> List[str]:
    """
    Sorts a list of control names by separating and ordering them based on specific suffixes.

    Parameters:
    lst (list[str]): A list of control names to be sorted.

    Returns:
    list[str]: A sorted list of control names with specific suffixes prioritized.
    """
    # Separate 'R' and 'L' strings
    r_strings  = [s for s in lst if "_R" in s]
    l_strings  = [s for s in lst if "_L" in s]
    ct_strings = [s for s in lst if "_Ct" in s]
    # Sort 'R' strings in descending order
    r_strings.sort()
    if l_strings:
        r_strings.sort(reverse=True)
    # Sort 'L' strings in ascending order
    l_strings.sort()

    # Combine the sorted lists
    rlist = r_strings + ct_strings + l_strings

    if not rlist:
        rlist = lst.copy()
        rlist.sort()
    # cull out mover names
    ignore_suffix = ["mover", "crv"]
    # rlist = [s for s in rlist if '_mover' not in s]
    remove = [s for s in rlist for suffix in ignore_suffix if suffix in s]
    rlist  = [s for s in rlist if s not in remove]

    return rlist


def get_roots(controls_list: List[str]) -> List[str]:
    """
    Extracts and returns the unique root names from a list of control names.

    Parameters:
    controls_list (list[str]): A list of control names, each containing a root name followed by an underscore and additional identifiers.

    Returns:
    list[str]: A sorted list of unique root names, sorted by length in descending order for easier string matching.
    """
    rlist = []
    for control in controls_list:
        root = control.split("_")[0]
        root = "".join([char for char in root if not char.isdigit()])
        rlist.append(root)

    rlist = list(set(rlist))
    # we want to sort by length and reverse the order for easier processing of string match
    return sorted(rlist, key=len, reverse=True)


def get_control_arrays(controls_list: List[str]) -> Dict[str, List[str]]:
    """
    Organizes a list of control names into a dictionary based on their root names.

    Parameters:
    controls_list (List[str]): A list of control names to be organized.

    Returns:
    Dict[str, List[str]]: A dictionary where keys are root names and values are lists of control names sorted by specific suffixes.
    """
    rdata   = {}
    roots   = get_roots(controls_list)
    removed = []
    for root in roots:
        for control in controls_list:
            if root in control:
                removed.append(control)
                if root not in rdata:
                    rdata[root] = []
                rdata[root].append(control)

    for root in rdata:
        rdata[root] = control_sort(rdata[root])

    return rdata


def retarget_function_data(function_data, bslpine_map_data, refactor_alignment=False):
    remapped = []
    for rig in bslpine_map_data:
        # find all the rig instances
        key_path_data = find_key(function_data, rig)
        if not key_path_data:
            print(f"No data found for rig: {rig}")
            continue
        bslpine_map        = bslpine_map_data[rig]
        refactor_init_data = bslpine_map["refactor_data"]
        remapped.append(rig)
        # retarget the pose data
        for key_data in key_path_data:
            keys_path = key_data[0]
            data      = key_data[1]
            # refactored_matrices = refactor_init_data["refactored_matrices"]
            data = refactor_control_object_matrices(
                data["world"], refactor_init_data, refactor_alignment=refactor_alignment
            )
            """
            data["world"] = refactor_control_object_matrices(data["world"],
                                                      refactor_init_data,
                                                      refactor_alignment=refactor_alignment)
            # now update the offset data
            data["offset"] = matrix_delta(refactored_matrices, data["world"])
            """
            # apply/replace the pose data
            set_value_by_path(function_data, keys_path, data)
    # cmds.select("GUS")


def batch_retarget_init(
    function_data:      dict,
    rig_args:           dict,
    mesh_data:          dict,
    uvn_scale:          Optional[List[float]] = None,
    refactor_alignment: bool                  = False,
) -> dict:
    """
    Initializes the retargeting process for a set of rigs, adjusting their function data to a new space.

    Parameters:
    function_data (dict): Contains the function data for the rigs, including control regions and initial data.
    rig_args (dict): Arguments for each rig, including spline and driven data.
    mesh_data (dict): Data for the mesh objects associated with each rig, including UV information.
    uvn_scale (Optional[List[float]]): Scale factors for U, V, and N directions. Defaults to [1.0, 1.0, 1.0].
    refactor_alignment (bool): Indicates whether to apply refactor alignment. Defaults to False.

    Returns:
    dict: A dictionary containing refactored function data, including region initialization, driven initialization, and B-spline mapping.
    """
    if not uvn_scale:
        uvn_scale = [1.0, 1.0, 1.0]
    # TODO clean up this function library from rigs that are not included in the maya sesssion
    function_library = copy.deepcopy(function_data["function_data"])

    function_refactor = {
        "rig_args":      rig_args,
        "region_init":   {},
        "driven_init":   {},
        "bslpine_map":   {},
        "function_data": function_library,
    }
    # this loop will refactor all the init data and function data into the new space.
    for rig in rig_args:
        # see if there is function data in the input
        if (
            rig not in function_data["control_regions"]
        ):  # we may eventually want reconciliation options here
            print(f"rig : {rig}  not found in input function data. Skipping!")
            continue
        function_refactor["region_init"][rig] = {
            "init_matrices": np.copy(rig_args[rig]["spline"]["matrices"])
        }
        function_refactor["driven_init"][rig] = copy.deepcopy(rig_args[rig]["driven"])

        refactor_inputs = [
            rig_args[rig]["spline"]["obj_names"],
            rig_args[rig]["spline"]["matrices"],
        ]
        control_inputs = [
            function_data["control_regions"][rig],
            function_data["region_init"][rig],
        ]
        driven_inputs = [
            rig_args[rig]["driven"]["obj_names"],
            rig_args[rig]["driven"]["matrices"],
        ]

        # get the retargeting matrix
        function_refactor["bslpine_map"][rig] = get_bspline_map_init(
            control_inputs,
            driven_inputs,
            Mesh=mesh_data[rig]["mesh_obj"],
            UVList=mesh_data[rig]["uv_list"],
            uv_map_index=mesh_data[rig]["uv_map_index"],
            uvn_scale=uvn_scale,
            refactor_inputs=refactor_inputs,
            refactor_alignment=refactor_alignment,
        )

    # now refactor all the function data for this rig
    retarget_function_data(function_library, function_refactor["bslpine_map"])

    return function_refactor


def get_function_pose(
    rig_function_data: dict, mesh_data: dict, pose: str, relative: bool = False
) -> dict:
    """
    Retrieves the pose data for a given rig and pose name, transforming it into the appropriate space.

    Parameters:
    rig_function_data (dict): Contains the function data for the rigs, including control regions and initial data.
    mesh_data (dict): Data for the mesh objects associated with each rig, including UV information.
    pose (str): The name of the pose to retrieve.
    relative (bool): Indicates whether to apply the pose relative to the initial matrices. Defaults to False.

    Returns:
    dict: A dictionary containing the transformed matrices and object names for each rig.
    """
    driven_pose_data = {}

    if not find_key(rig_function_data["function_data"], pose):
        print(f" pose: {pose}  not forund in function_data")
        return

    pose_data = find_key(rig_function_data["function_data"], pose)[0][1]

    for rig in rig_function_data["bslpine_map"]:
        driven       = rig_function_data["bslpine_map"][rig]["driven"]
        bspline_map  = rig_function_data["bslpine_map"][rig]

        mesh_obj     = mesh_data[rig]["mesh_obj"]
        uv_list      = mesh_data[rig]["uv_list"]
        uv_map_index = mesh_data[rig]["uv_map_index"]

        # get init matrices of bspline
        init_matrices = rig_function_data["region_init"][rig]["init_matrices"]

        # get bspline pose in spaces
        # print(pose_data[rig]['offset'])
        pose_matrices   = np.copy(pose_data[rig]["world"])
        offset_matrices = pose_data[rig]["offset"]

        if relative:
            init_points = control_array_type_conversion(init_matrices)
            # print(init_points)
            init_matrices = convert_points_to_surfacespace_matrices(
                init_points, Mesh=mesh_obj, UVList=uv_list, uv_map_index=uv_map_index
            )
            pose_matrices = matrix_multiply(offset_matrices, init_matrices)

        control_data = {
            "matrices": pose_matrices,
            "points":   control_array_type_conversion(pose_matrices),
        }
        # weigh the driven
        bspline_result = bspline_weigh_transformations(
            control_data,
            bspline_map,
            Mesh            = mesh_obj,
            UVList          = uv_list,
            uv_map_index    = uv_map_index,
            normal_scale    = 1.0,
            return_matrices = True,
        )

        driven_pose_data[rig] = {
            "matrices":  bspline_result["transformation"],
            "obj_names": driven,
        }

    return driven_pose_data


def convert_dict_vals_arrays(dictionary: dict, to_numpy_array: bool = False) -> dict:
    """
    Recursively iterate over a dictionary and convert any numpy arrays to lists.
    Args:
        dictionary (dict): The dictionary to process.
        to_numpy_array (bool): Flag to determine conversion direction.
    Returns:
        dict: A new dictionary with the same structure, but with numpy arrays replaced by lists.
    """
    # Create a new dictionary to store the results
    result = {}

    change_type = np.ndarray
    if to_numpy_array:
        change_type = list

    # Iterate over each key-value pair in the input dictionary
    for key, value in dictionary.items():
        # If the value is a dictionary, recursively process it
        if isinstance(value, dict):
            result[key] = convert_dict_vals_arrays(value, to_numpy_array)
        # If the value is a numpy array, convert it to a list
        elif isinstance(value, change_type):
            if to_numpy_array:
                if not isinstance(value[0], str):
                    result[key] = np.array(value)
            else:
                result[key] = value.tolist()
        # Otherwise, just copy the value into the result dictionary
        else:
            result[key] = value
    return result