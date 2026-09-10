# `cgmath` Reference

Generated from the live code by `gen_reference.py` (kept in the `tools/` folder beside the packages, not in this repository) on 2026-09-10. Do not edit by hand; re-run the script after any API change.

Every public class, property, method, function and constant of the modules below, with the real signature (`inspect.signature`) and the first line of its docstring. To *learn* a subpackage read its `README.md` and `CHEATSHEET.md`; this page is for looking things up.

Conventions: matrices are row-major with translation at `M[3, :3]`; quaternions are `(i, j, k, w)`; the free functions of the separate `transforms` package take radians while node channels are degrees; rotate orders `XYZ=0 YZX=1 ZXY=2 XZY=3 YXZ=4 ZYX=5`.

---

## Index

- [`cgmath.utils`](#cgmathutils)
- [`cgmath.hierarchy`](#cgmathhierarchy)
- [`cgmath.geometry`](#cgmathgeometry)
- [`cgmath.geometry._base`](#cgmathgeometry_base)
- [`cgmath.geometry.mesh`](#cgmathgeometrymesh)
- [`cgmath.geometry.map`](#cgmathgeometrymap)
- [`cgmath.geometry.morph_target`](#cgmathgeometrymorph_target)
- [`cgmath.geometry.skin_weights`](#cgmathgeometryskin_weights)
- [`cgmath.geometry.bspline`](#cgmathgeometrybspline)
- [`cgmath.geometry.bspline_patch`](#cgmathgeometrybspline_patch)
- [`cgmath.geometry._saddle_surface`](#cgmathgeometry_saddle_surface)
- [`cgmath.geometry._cdt`](#cgmathgeometry_cdt)
- [`cgmath.geometry.sdf`](#cgmathgeometrysdf)
- [`cgmath.geometry.pack`](#cgmathgeometrypack)
- [`cgmath.geometry.resample`](#cgmathgeometryresample)
- [`cgmath.geometry.robust_skinweights_transfer_bilinear`](#cgmathgeometryrobust_skinweights_transfer_bilinear)
- [`cgmath.geometry.surface_plotting`](#cgmathgeometrysurface_plotting)
- [`cgmath.geometry.deform`](#cgmathgeometrydeform)
- [`cgmath.geometry.utils`](#cgmathgeometryutils)
- [`cgmath.render`](#cgmathrender)
- [`cgmath.render.scene`](#cgmathrenderscene)
- [`cgmath.render.frame`](#cgmathrenderframe)
- [`cgmath.render.camera`](#cgmathrendercamera)
- [`cgmath.render.raytracer`](#cgmathrenderraytracer)
- [`cgmath.render.texture`](#cgmathrendertexture)
- [`cgmath.formats.glb`](#cgmathformatsglb)
- [`cgmath.formats.fbx`](#cgmathformatsfbx)
- [`cgmath.formats.usd.stage`](#cgmathformatsusdstage)
- [`cgmath.formats.usd.prim`](#cgmathformatsusdprim)
- [`cgmath.rbf._kernels`](#cgmathrbf_kernels)
- [`cgmath.constraints`](#cgmathconstraints)

---

## `cgmath.utils`

docstring access, timing, pretty JSON, the test runner.

#### `json`

Provides a json module-like interface to read/write pretty_json

```text
json()
```

Methods:

```text
staticmethod dump(obj: 'list | dict | tuple | np.ndarray', fp: 'IO[str]', indent: 'int' = 4, max_line_width: 'int' = 80) -> 'None'
staticmethod dumps(obj: 'list | dict | tuple | np.ndarray', indent: 'int' = 4, max_line_width: 'int' = 80) -> 'str'
staticmethod load(*args, **kwargs)
staticmethod loads(*args, **kwargs)
```

Functions:

```text
docstring(obj: 'Any') -> 'str'
    Return the docstring of any Python object as a plain string.
info(obj: 'Any') -> 'str'
    Return the docstring of any Python object as a plain string.
pretty_json(data: 'dict | list | tuple', indent: 'int' = 4, max_line_width: 'int' = 80) -> 'str'
    Serialize data to human-readable JSON with columnar array formatting.
profile(cmd: 'str', n: 'int' = 1) -> 'float'
    simple speed profiler
run_tests(target: 'str | Sequence[str] | None' = None, verbosity: 'int' = 2, failfast: 'bool' = False) -> 'unittest.TestResult'
    runs the package's unit test suite
```

## `cgmath.hierarchy`

transform nodes, hierarchies and clips (implemented in `hierarchy/hierarchy.py`).

> Transform nodes, hierarchies and animation clips.

Re-exported classes:

```text
ClipData
HierarchyData
TransformData
TransformList
```

#### `ClipData(HierarchyData)`

A HierarchyData holding F poses rather than one.

```text
ClipData(iterable=None, frames: 'Optional[int]' = None, start_frame: 'int' = 0, fps: 'float' = 24.0)
```

Properties:

```text
end_frame                          
fps                                
frame                              
frame_count                        
frames                             
start_frame                        
```

Methods:

```text
add_delta(self, delta, by: 'str' = 'index', translate: 'bool' = True) -> "'ClipData'"
    Every frame with ``delta`` applied, as a clip.
copy(self) -> "'ClipData'"
    returns a deep copy, owning the nodes it holds.
classmethod from_dict(cls, data: 'dict') -> "'ClipData'"
classmethod from_poses(cls, poses, start_frame: 'int' = 0, fps: 'float' = 24.0) -> "'ClipData'"
    builds a clip from a sequence of HierarchyData poses
get_delta(self, other, by: 'str' = 'index', translate: 'bool' = True) -> "'ClipData'"
    The delta of every frame against ``other``, as a clip.
classmethod load_fbx(cls, filename: 'str', scale_factor: 'float' = 1.0, take: 'Optional[int]' = None, fps: 'Optional[float]' = None, start_frame: 'Optional[int]' = None, end_frame: 'Optional[int]' = None) -> "'ClipData'"
    builds a clip from an fbx's animation take
classmethod load_glb(cls, filename: 'str', scale_factor: 'float' = 100.0, animation: 'int' = 0, fps: 'float' = 24.0, start_frame: 'int' = 0) -> "'ClipData'"
    builds a clip from a glb's animation
save_fbx(self, filename, zero_root=False)
    saves the bound frame to a .fbx file
to_dict(self) -> 'dict'
    set keys as uuid in the case of duplicate names
```

#### `HierarchyData(TransformList)`

Owning container for a hierarchy of ``TransformData`` nodes.

```text
HierarchyData(iterable=None)
```

Methods:

```text
append(self, node: "'TransformData'") -> 'None'
    S.append(value) -- append value to the end of the sequence
insert(self, index: 'int', node: "'TransformData'") -> 'None'
    S.insert(index, value) -- insert value before index
pop(self, index=-1)
    S.pop([index]) -> item -- remove and return item at index (default last).
```

#### `TransformData(Data)`

TransformData(name: 'str', node_type: 'Optional[str]' = 'transform', uuid: 'Optional[str]' = None, parent_node: 'Optional[str]' = None, scale: 'Optional[np.ndarray]' = None, rotate: 'Optional[np.ndarray]' = None, translate: 'Optional[np.ndarray]' = None, rotate_order: 'Optional[int]' = None, rotate_axis: 'Optional[np.ndarray]' = None, joint_orient: 'Optional[np.ndarray]' = None, visibility: 'Optional[bool]' = None, segment_scale_compensate: 'Optional[bool]' = None, radius: 'Optional[float]' = None, draw_style: 'Optional[int]' = None, user_defined_attributes: 'Optional[dict]' = None)

```text
TransformData(name: 'str', node_type: 'Optional[str]' = 'transform', uuid: 'Optional[str]' = None, parent_node: 'Optional[str]' = None, scale: 'Optional[np.ndarray]' = None, rotate: 'Optional[np.ndarray]' = None, translate: 'Optional[np.ndarray]' = None, rotate_order: 'Optional[int]' = None, rotate_axis: 'Optional[np.ndarray]' = None, joint_orient: 'Optional[np.ndarray]' = None, visibility: 'Optional[bool]' = None, segment_scale_compensate: 'Optional[bool]' = None, radius: 'Optional[float]' = None, draw_style: 'Optional[int]' = None, user_defined_attributes: 'Optional[dict]' = None)
```

Fields:

```text
name: str = None
node_type: Optional[str] = 'transform'
uuid: Optional[str] = None
parent_node: Optional[str] = None
user_defined_attributes: Optional[dict] = None
```

Properties:

```text
draw_style                         
index                              return's the node's index in the assigned hierarchy
joint_orient                       
joint_orient_matrix                
joint_orient_x                     
joint_orient_y                     
joint_orient_z                     
matrix                             computes the local transform matrix
parent_scale_inverse               
parent_scale_inverse_matrix        
quaternion                         rotation as a quaterion
radius                             
rotate                             
rotate_axes                        returns the rotate order as a string
rotate_axis                        
rotate_axis_matrix                 
rotate_axis_x                      
rotate_axis_y                      
rotate_axis_z                      
rotate_matrix                      
rotate_order                       
rotate_x                           
rotate_y                           
rotate_z                           
scale                              
scale_matrix                       
scale_x                            
scale_y                            
scale_z                            
segment_scale_compensate           
translate                          
translate_matrix                   
translate_x                        
translate_y                        
translate_z                        
unique_name                        returns True if no other node has this name
visibility                         
world_matrix                       compute a world matrix
```

Methods:

```text
add_prefix(self, prefix: 'str') -> 'None'
    adds a prefix to the node's name
add_suffix(self, suffix: 'str') -> 'None'
    adds a suffix to the node's name
get_branch(self) -> "'TransformList'"
    returns a recursive NodeList from of all decendents
get_children(self) -> "'TransformList'"
    returns the node's children
get_parent(self) -> "Union['TransformData', None]"
    returns this TransformData's parent TransformData
get_parent_matrix(self) -> 'np.ndarray'
    returns the parent's worldspace matrix
get_root(self) -> "'TransformList'"
    returns a NodeList of the branch from root to self
match_matrix(self, other: "'TransformData'")
    matches the world matrix of a TransformData object
match_rotate(self, other: "'TransformData' | np.ndarray")
    matches the world rotation of a TransformData object
match_translate(self, other: "'TransformData' | np.ndarray", x: 'bool' = True, y: 'bool' = True, z: 'bool' = True)
    matches the world position of a TransformData object
set_joint_orient_to_rotate(self) -> 'None'
    sets joint_orient and rotate_axis to 0 and applies all rotation to rotate
set_parent(self, parent: 'Union[str, None]', world_space: 'bool' = True) -> 'None'
    reparents node to a given parent and resets internal SRTs
set_rotate_to_joint_orient(self) -> 'None'
    sets rotate and rotate_axis to 0 and applies all rotation to joint_orient
swapaxes(self, axis0: 'int', axis1: 'int', negate: 'bool' = False) -> 'None'
    swaps two of this node's local axes without moving its children
to_attributes(self, mapping: 'dict' = {'_scale': 'scale', '_rotate': 'rotate', '_translate': 'translate', '_rotate_order': 'rotateOrder', '_rotate_axis': 'rotateAxis', '_joint_orient': 'jointOrient', '_segment_scale_compensate': 'segmentScaleCompensate', '_radius': 'radius', '_visibility': 'visibility', '_draw_style': 'drawStyle'}) -> 'dict'
    returns a dict mapped against expected DCC node attributes
```

#### `TransformList(DataList)`

Non-owning ordered view over ``TransformData`` nodes.

```text
TransformList(iterable=None)
```

Properties:

```text
draw_style                         
indices                            returns each node's index in its owning hierarchy
joint_orient                       
joint_orient_matrix                
joint_orient_x                     
joint_orient_y                     
joint_orient_z                     
matrix                             
name                               returns a list of all names
node_type                          
parent_scale_inverse               
parent_scale_inverse_matrix        
quaternion                         
radius                             
rotate                             
rotate_axes                        returns each node's rotate order as a string
rotate_axis                        
rotate_axis_matrix                 
rotate_axis_x                      
rotate_axis_y                      
rotate_axis_z                      
rotate_matrix                      
rotate_order                       
rotate_x                           
rotate_y                           
rotate_z                           
scale                              
scale_matrix                       
scale_x                            
scale_y                            
scale_z                            
segment_scale_compensate           
translate                          
translate_matrix                   
translate_x                        
translate_y                        
translate_z                        
unique_name                        
uuid                               returns a list of all node uuids
visibility                         
world_matrix                       
```

Methods:

```text
add_delta(self, delta: "'HierarchyData'", by: 'str' = 'index', translate: 'bool' = True) -> "'HierarchyData'"
    Return ``self`` with ``delta`` applied, as a new hierarchy.
add_prefix(self, prefix: 'str') -> 'None'
    adds a prefix to the node's name
add_suffix(self, suffix: 'str') -> 'None'
    adds a suffix to the node's name
append(self, node: "'TransformData'") -> 'None'
    S.append(value) -- append value to the end of the sequence
copy(self) -> "'TransformList'"
    returns a deep copy, owning the nodes it holds.
get_branch(self) -> "'TransformList'"
    returns every node in the view and all of their decendents
get_children(self) -> "'TransformList'"
    returns the children of every node in the view, without duplicates
get_delta(self, other, by: 'str' = 'index', translate: 'bool' = True) -> "'HierarchyData'"
    Return the delta ``self - other``, ie: what ``other`` must gain to
get_parent_matrix(self) -> 'np.ndarray'
    returns every node's parent worldspace matrix
get_parents(self) -> 'np.ndarray'
    returns the node's parent index
get_roots(self) -> "'TransformList'"
    returns all contained nodes parented to the world
index(self, obj, start=0, stop=None) -> 'int'
    S.index(value, [start, [stop]]) -> integer -- return first index of value.
insert(self, index: 'int', node: "'TransformData'") -> 'None'
    S.insert(index, value) -- insert value before index
classmethod load(cls, filename: 'str', mode: 'Optional[str]' = None)
    loads the data from a file
classmethod load_fbx(cls, filename: 'str', scale_factor: 'float' = 1.0)
    primitive fbx joint loader
classmethod load_glb(cls, filename: 'str', scale_factor: 'float' = 100.0, pose_frame: 'None | int' = None)
    primitive glb joint loader that supports style2 and momentum glbs
match(self, *args, exclude: 'bool' = False, exact: 'bool' = True) -> "'TransformList'"
    returns a TransformList object with matching names
match_matrix(self, other) -> 'None'
    matches the world matrix of a node or a list of nodes
match_rotate(self, other) -> 'None'
    matches the world rotation of a node, a list of nodes or a rotation
match_translate(self, other, x: 'bool' = True, y: 'bool' = True, z: 'bool' = True) -> 'None'
    matches the world position of a node, a list of nodes or a translation
save_fbx(self, filename, zero_root=False)
    saves the data to a .fbx file
set_joint_orient_to_rotate(self) -> 'None'
    sets joint_orient to 0 and applies all rotation to rotate
set_parent(self, parent: 'Union[str, None]', world_space: 'bool' = True) -> 'None'
    sets all the nodes under a given parent
set_rotate_to_joint_orient(self) -> 'None'
    sets rotate to 0 and applies all rotation to joint_orient
swapaxes(self, axis0: 'int', axis1: 'int', negate: 'bool' = False) -> 'None'
    swaps two local axes on every node in the view
to_attributes(self, mapping: 'dict' = {'_scale': 'scale', '_rotate': 'rotate', '_translate': 'translate', '_rotate_order': 'rotateOrder', '_rotate_axis': 'rotateAxis', '_joint_orient': 'jointOrient', '_segment_scale_compensate': 'segmentScaleCompensate', '_radius': 'radius', '_visibility': 'visibility', '_draw_style': 'drawStyle'}) -> 'List[dict]'
    returns every node's attributes as a list of dicts
to_dict(self) -> 'dict'
    set keys as uuid in the case of duplicate names
```

Functions:

```text
generate_uuid() -> 'str'
    generates a new unique uuid
validate_uuid(uuid_string: 'str') -> 'bool'
    validates a uuid string
```

Constants:

```text
MAYA_ATTRIBUTE_MAP = {'_scale': 'scale', '_rotate': 'rotate', '_translate': 't...
```

## `cgmath.geometry`

the re-exported public surface of the geometry package.

> Move all data types into the same namespace

Re-exported classes:

```text
BSplineData
BSplinePatchData
CompactSkinData
Data
DataList
DeltaMushData
GeomSubsetData
ImmutableArray
MapData
MeshData
MeshList
MorphData
MorphList
PatchRelaxData
PatchSampleData
RaycastData
SkinData
SkinList
TriangulateMethod
TriangulateRules
UVData
UVList
```

## `cgmath.geometry._base`

the `Data` / `DataList` contract every type is built on.

#### `Data`

Base class with managed serialization.

```text
Data() -> None
```

Methods:

```text
copy(self)
    returns a deep copy of self
classmethod from_bytes(cls, data: bytes) -> Any
classmethod from_dict(cls, data: dict) -> Any
classmethod info(cls) -> str
    Returns this class's docstring as a string.
classmethod load(cls, filename: str, mode: Optional[str] = None) -> Any
    loads the data from a file
classmethod load_json(cls, filename: str) -> Any
classmethod load_npz(cls, filename: str) -> Any
classmethod load_pickle(cls, filename: str) -> Any
match(self, *args, exact: bool = True) -> bool
    returns True if name matches any arguments
reset_cached_data(self)
    resets cached data
save(self, filename: str, mode: Optional[str] = None) -> str
    saves the data to a file
save_json(self, filename: str) -> str
save_npz(self, filename: str, compression=8) -> str
save_pickle(self, filename: str) -> str
to_bytes(self) -> bytes
    returns a bytes object from the zipped data
to_dict(self) -> dict
    returns the annotated data as a dict
to_json(self) -> str
    formats the data as human readable json
```

#### `DataList(MutableSequence)`

All the operations on a read-write sequence.

```text
DataList(iterable=None)
```

Methods:

```text
append(self, value)
    S.append(value) -- append value to the end of the sequence
copy(self)
    deep copy
extend(self, iterable)
    S.extend(iterable) -- extend sequence by appending elements from the iterable
classmethod from_bytes(cls, data: bytes) -> Any
classmethod from_dict(cls, data: dict) -> Any
index(self, value, start=0, stop=None)
    S.index(value, [start, [stop]]) -> integer -- return first index of value.
insert(self, index, value)
    S.insert(index, value) -- insert value before index
classmethod load(cls, filename: str, mode: Optional[str] = None) -> Any
    loads the data from a file
classmethod load_json(cls, filename: str) -> Any
classmethod load_npz(cls, filename: str) -> Any
classmethod load_pickle(cls, filename: str) -> Any
match(self, *args, exclude: bool = False, exact: bool = True) -> Any
    returns a DataList object with matching names
pop(self, index=-1)
    S.pop([index]) -> item -- remove and return item at index (default last).
reset_cached_data(self)
    resets cached data
save(self, filename: str, mode: Optional[str] = None) -> str
    saves the data to a file
save_json(self, filename: str) -> str
    saves the morph target list to a json file
save_npz(self, filename: str) -> str
    saves the morph target list to a npz file
save_pickle(self, filename: str) -> str
sort(self, key=None, reverse=False)
to_bytes(self) -> bytes
    returns a bytes object from the zipped data
to_dict(self) -> dict
to_json(self) -> str
    formats the data as human readable json
```

#### `ImmutableArray(ndarray)`

A hack used to bypass a python 3.10+ limitation with dataclasses fields and numpy arrays.

```text
ImmutableArray(input_array)
```

Functions:

```text
bytes_to_dict(data)
    takes a bytes object and returns a deep dict of numpy arrays
dict_to_bytes(data)
    takes a deep dict and returns a zip file as bytes
dict_to_zip(data, filename, compression=8)
    takes a deep dict and saves a zip file with .npy data
flatten_nested_lists(xs)
get_annotations(cls)
    returns a dict of all annotations including inherited ones
get_file_type(filename: str) -> str
    identifies the supported file type by reading the header
is_ndarray_annotation(dtype) -> bool
    returns True if an annotation is np.ndarray, or a union holding it
nan_to_none(obj)
    convert np.nan (not a number) to None object
none_to_nan(obj)
    convert None object to np.nan (not a number)
zip_to_dict(filename)
    takes a zip file and returns a deep dict of numpy arrays
```

Constants:

```text
MISSING = <dataclasses._MISSING_TYPE object at 0x000001C2B3719B10>
```

## `cgmath.geometry.mesh`

meshes, UVs, triangulation rules, loaders.

#### `AreaMethod(Enum)`

Create a collection of name/value pairs.

```text
AreaMethod(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)
```

#### `Axis(Enum)`

Create a collection of name/value pairs.

```text
Axis(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)
```

#### `MeshData(Data)`

Polygonal descriptor representing an object with edges, faces and vertices.

```text
MeshData(indices: 'np.ndarray', counts: 'np.ndarray', points: 'np.ndarray', name: 'Optional[str]' = None, matrix: 'Optional[np.ndarray]' = ImmutableArray([[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]]), hole_faces: 'Optional[np.ndarray]' = None, hole_counts: 'Optional[np.ndarray]' = None, hole_indices: 'Optional[np.ndarray]' = None, normals: 'Optional[np.ndarray]' = None, normal_indices: 'Optional[np.ndarray]' = None) -> None
```

Fields:

```text
indices: np.ndarray
counts: np.ndarray
points: np.ndarray
name: Optional[str] = None
matrix: Optional[np.ndarray] = ImmutableArray([[1., 0., 0., 0.],
      
hole_faces: Optional[np.ndarray] = None
hole_counts: Optional[np.ndarray] = None
hole_indices: Optional[np.ndarray] = None
normals: Optional[np.ndarray] = None
normal_indices: Optional[np.ndarray] = None
```

Properties:

```text
area                               
bvh_bezier                         Lazily-built BVH over PN-Quad bicubic Bezier control-point AABBs.
bvh_bilinear                       Lazily-built BVH over per-face vertex AABBs.
closed                             returns True if mesh has no border edges
e2f                                
e2v                                
edge_count                         returns the mesh's face count
f2e                                
f2ue                               
f2v                                
face_count                         returns the mesh's face count
geometry                           returns what you'd consider the geometric construct
has_degenerate_faces               returns True if any face cannot be a polygon
ngons                              returns the ngon count for the mesh
open                               returns True if mesh has border edges
point_count                        returns the mesh's vertex count
points4                            returns self.points as a 4d sequence for matrix math purposes
quads                              returns the quad count for the mesh
shell_count                        returns the number of shells
shell_edges                        returns the shells as edge indices
shell_faces                        returns the shells as face indices
shell_points                       returns the shells as point indices
triangles                          returns the triangle count for the mesh
ue                                 
ue2f                               
ue2v                               
uep                                
v2e                                
v2f                                
v2ue                               
valence                            returns the max valence for the mesh vertices
valid                              
```

Methods:

```text
apply_morph_target(self, target_data) -> 'None'
    Applies a morph target to this mesh.
contains_vertices(self, vertices, contained=True, exclude=False)
    returns the mesh's faces which contains the given vertex indices
delete_faces(self, faces, exclude=False)
    deletes faces at given face indices
detach_faces(self, indices=None)
    detaches faces
difference(self, *args, tolerance=1e-06, world_space=False)
    removes overlapping faces of given Mesh objects from self (this is not a boolean)
edges_to_vertex_pairs(self, edges: 'np.ndarray') -> 'np.ndarray'
    returns the vertex pairs for a given set of edges
edges_to_vertices(self, edges: 'np.ndarray') -> 'np.ndarray'
    returns the vertices for a given set of edges
faces_to_vertices(self, faces: 'np.ndarray') -> 'np.ndarray'
    returns the vertices for a given set of faces
fix_symmetry(self, pivot: 'float' = 0.0, axis: 'int' = 0, side: 'float' = 1.0, tolerance: 'Union[None, float]' = None, force: 'bool' = False) -> 'bool'
    fixes the mesh point symmetry, return True/False if the result is symmetrical
from_faces(self, faces, exclude=False)
    rebuilds a new mesh from given face indices
classmethod from_prim(cls, prim: 'Any') -> "'MeshData'"
    Construct a data object from a mesh prim.
from_vertices(self, vertices, contained=True, exclude=False)
    rebuilds a new mesh from given vertex indices
get_asymmetric_points(self, pivot: 'float' = 0.0, axis: 'int' = 0, tolerance: 'Union[None, float]' = None)
    returns the vertex indices that fail the symmetry test
get_border_edges(self, flatten=False)
    returns all border edges
get_border_faces(self, flatten=False)
    returns all border faces
get_border_vertices(self, flatten=False)
    returns all border vertices
get_center_points(self, pivot: 'float' = 0.0, axis: 'int' = 0, tolerance: 'Union[None, float]' = None) -> 'np.ndarray'
    returns the indices of the center points
get_closest_points(self, points: 'np.ndarray', k: 'int' = 1) -> 'Tuple[np.ndarray, np.ndarray]'
    Returns the distances and indices of the closest points to the given points.
get_correspondence(self, mesh: "'MeshData'") -> 'None'
    computes the point correspondence between MeshData objects
get_degenerate_faces(self)
    returns the face indices that cannot be a polygon
get_edge_face_clusters(self, indices: 'np.ndarray') -> 'List[np.ndarray]'
    convert edge face indices to clusters
get_edge_face_geodesic_neighborhood(self, n: 'int | None' = None, max_distance: 'float | None' = None, indices: 'np.ndarray | None' = None, distance_format='sparse')
    Compute the topological edge face neighborhood with distances using Dijkstra's algorithm
get_edge_face_neighbors(self, n: 'int' = 0) -> 'np.ndarray'
    return a matrix of faces neighboring connected edges to faces
get_edge_lengths(self) -> 'np.ndarray'
    Returns the edge lengths.
get_edge_normals(self) -> 'np.ndarray'
get_edge_vertex_clusters(self, indices: 'np.ndarray') -> 'List[np.ndarray]'
    convert edge vertex indices to clusters
get_edge_vertex_geodesic_neighborhood(self, n: 'int | None' = None, max_distance: 'float | None' = None, indices: 'np.ndarray | None' = None, distance_format='sparse')
    Compute the topological edge vertex neighborhood with distances using Dijkstra's algorithm
get_edge_vertex_neighbors(self, n: 'int' = 0) -> 'np.ndarray'
    return a matrix of vertices neighboring connected edges to vertices
get_extent(self, per_face: 'bool' = False) -> 'Tuple[np.ndarray]'
    Returns 2 tuples that represents the extent of this mesh.
get_face_areas(self, method: 'AreaMethod | str' = <AreaMethod.GAUSS: 'gauss'>, samples: 'int' = 10, tolerance: 'float' = 0.0001)
    returns a face area approximation
get_face_edge_clusters(self, indices: 'np.ndarray') -> 'List[np.ndarray]'
    convert face edge indices to clusters
get_face_edge_geodesic_neighborhood(self, n: 'int | None' = None, max_distance: 'float | None' = None, indices: 'np.ndarray | None' = None, distance_format='sparse')
    Compute the topological face edge neighborhood with distances using Dijkstra's algorithm
get_face_edge_neighbors(self, n: 'int' = 0) -> 'np.ndarray'
    return a matrix of edges neighboring connected faces to edges
get_face_normals(self, force: 'bool' = False) -> 'np.ndarray'
    Returns the face normals. Computes it if not already computed.
get_face_vertex_angles(self, force: 'bool' = False) -> 'np.ndarray'
    Compute vertex angles in radians as a per face per vertex array
get_face_vertex_clusters(self, indices: 'np.ndarray') -> 'List[np.ndarray]'
    convert face vertex indices to clusters
get_face_vertex_geodesic_neighborhood(self, n: 'int | None' = None, max_distance: 'float | None' = None, indices: 'np.ndarray | None' = None, distance_format='sparse')
    Compute the topological face vertex neighborhood with distances using Dijkstra's algorithm
get_face_vertex_neighbors(self, n: 'int' = 0) -> 'np.ndarray'
    return a matrix of vertices neighboring connected faces to vertices
get_inverse_matrix(self) -> 'np.ndarray'
    computes the inverse matrix
get_lamina_faces(self)
    returns lamina faces
get_ngons(self)
    returns the face ngon count
get_non_manifold_vertices(self)
    returns non-manifold vertices
get_overlap_vertices(self, tolerance=1e-06)
    returns overlapping pairs of vertices
get_quadrangulate_rules(self, planarity_weight: 'float' = 1.0, squareness_weight: 'float' = 0.5, diagonal_weight: 'float' = 0.5, min_score: 'float' = 0.0, require_convex: 'bool' = True) -> 'np.ndarray'
    Identifies pairs of triangles to merge back into quads.  This is
get_quads(self)
    returns the face indices that are quads
get_subset_rules(self, vertices, exclude=False, min_count=3)
    returns the local vertex slots each face keeps for a vertex subset
get_symmetry_deviation(self, pivot: 'float' = 0.0, axis: 'int' = 0) -> 'np.ndarray'
    returns the min/max distance in the symmetry mapping
get_symmetry_map(self, pivot: 'float' = 0.0, axis: 'int' = 0) -> 'Tuple[np.ndarray, np.ndarray]'
    returns the symmetric mapping for each vertex
get_tangent_space(self, uvdata: "'UVData'") -> 'Tuple[np.ndarray, np.ndarray]'
    Compute MikkTSpace tangent vectors.
get_triangles(self)
    returns the face indices that are triangles
get_triangulate_rules(self, method: 'TriangulateMethod | str' = <TriangulateMethod.FAST: 'fast'>, invert: 'bool' = False, ngons_only: 'bool' = False) -> 'TriangulateRules'
    generates triangulation rules according to specified method
get_unused_points(self)
    returns indices of unused points
get_valence(self)
    returns the number of edges connected to a vertex
get_vertex_edge_clusters(self, indices: 'np.ndarray') -> 'List[np.ndarray]'
    convert vertex edge indices to clusters
get_vertex_edge_geodesic_neighborhood(self, n: 'int | None' = None, max_distance: 'float | None' = None, indices: 'np.ndarray | None' = None, distance_format='sparse')
    Compute the topological vertex edge neighborhood with distances using Dijkstra's algorithm
get_vertex_edge_neighbors(self, n: 'int' = 0) -> 'np.ndarray'
    return a matrix of edges neighboring connected vertices to edges
get_vertex_face_clusters(self, indices: 'np.ndarray') -> 'List[np.ndarray]'
    convert vertex face indices to clusters
get_vertex_face_geodesic_neighborhood(self, n: 'int | None' = None, max_distance: 'float | None' = None, indices: 'np.ndarray | None' = None, distance_format='sparse')
    Compute the topological vertex face neighborhood with distances using Dijkstra's algorithm
get_vertex_face_neighbors(self, n: 'int' = 0) -> 'np.ndarray'
    return a matrix of faces neighboring connected vertices to faces
get_vertex_normals(self, angle_weighted: 'bool' = True, area_weighted: 'bool' = True, force: 'bool' = False, _areas: 'np.ndarray | None' = None) -> 'np.ndarray'
    Computes vertex normals from face normals.
get_zero_area_faces(self, tolerance=1e-12)
    returns the face indices enclosing no area
grow_edge_face_indices(self, indices: 'np.ndarray', n: 'int' = 1) -> 'np.ndarray'
    grows edge face indices by n steps
grow_edge_vertex_indices(self, indices: 'np.ndarray', n: 'int' = 1) -> 'np.ndarray'
    grows edge vertex indices by n steps
grow_face_edge_indices(self, indices: 'np.ndarray', n: 'int' = 1) -> 'np.ndarray'
    grows face edge indices by n steps
grow_face_vertex_indices(self, indices: 'np.ndarray', n: 'int' = 1) -> 'np.ndarray'
    grows face vertex indices by n steps
grow_vertex_edge_indices(self, indices: 'np.ndarray', n: 'int' = 1) -> 'np.ndarray'
    grows vertex edge indices by n steps
grow_vertex_face_indices(self, indices: 'np.ndarray', n: 'int' = 1) -> 'np.ndarray'
    grows vertex face indices by n steps
classmethod load_fbx(cls, filename: 'str', name: 'str | None' = None) -> "'MeshData'"
    returns a single MeshData from an fbx file (first mesh if name is None)
classmethod load_glb(cls, filename: 'str', scale_factor: 'float' = 100.0) -> "'MeshData'"
    a basic glb file reader to MeshData
classmethod load_obj(cls, filename: 'str') -> "'MeshData'"
    a basic obj file reader to MeshData
merge(self)
    merges overlapping points
quadrangulate(self, rules: 'np.ndarray | None' = None) -> 'None'
    Merges pairs of triangles into quads, in-place.
raycast(self, origins: 'MeshData | np.ndarray', directions: 'np.ndarray | None' = None, method: 'SampleMethod | str' = <SampleMethod.BILINEAR: 'bilinear'>, forward_only: 'bool' = True, twosided: 'bool' = True) -> "'RaycastData'"
    ray-mesh intersection on bilinear or bicubic Bezier patches
recompute_normals(self, angle_weighted: 'bool' = True, area_weighted: 'bool' = True) -> 'None'
    Recompute shading normals from current positions.
rules_to_vertices(self, rules)
    returns the vertices a set of subset rules keeps
sample(self, obj: "Union['MeshData', np.ndarray]", method: 'SampleMethod | str' = <SampleMethod.BILINEAR: 'bilinear'>, iteration_count: 'int' = 100, iteration_tolerance: 'float' = 1e-08)
    a humble saddle surface sampler
save_obj(self, filename: 'str') -> 'None'
    writes a valid obj file from MeshData
set_normals(self, hard_edge_angle: 'float | None' = None, hard_edges: 'np.ndarray | None' = None, angle_weighted: 'bool' = True, area_weighted: 'bool' = True) -> 'None'
    Compute and store face-varying shading normals.
shrink_edge_face_indices(self, indices: 'np.ndarray', n: 'int' = 1) -> 'np.ndarray'
    shrinks vertex face indices by n steps
shrink_edge_vertex_indices(self, indices: 'np.ndarray', n: 'int' = 1) -> 'np.ndarray'
    shrinks edge vertex indices by n steps
shrink_face_edge_indices(self, indices: 'np.ndarray', n: 'int' = 1) -> 'np.ndarray'
    shrinks face edge indices by n steps
shrink_face_vertex_indices(self, indices: 'np.ndarray', n: 'int' = 1) -> 'np.ndarray'
    shrinks face vertex indices by n steps
shrink_vertex_edge_indices(self, indices: 'np.ndarray', n: 'int' = 1) -> 'np.ndarray'
    shrinks vertex edge indices by n steps
shrink_vertex_face_indices(self, indices: 'np.ndarray', n: 'int' = 1) -> 'np.ndarray'
    shrinks vertex face indices by n steps
smooth(self, neighbors: 'np.ndarray | None' = None, indices: 'np.ndarray | None' = None, iterations: 'int | np.ndarray' = 1, receptions: 'float | np.ndarray' = 0.5, contributions: 'float | np.ndarray' = 1.0)
subdivide(self, steps: 'int' = 1, keep_borders: 'bool' = False, keep_edges: 'bool' = False, fit: 'bool' = False, tolerance: 'float' = 0.001, iterations: 'int' = 10000, damping: 'float' = 0.5) -> 'None'
sum_vertex_normals(self, face_vertex_normals: 'np.ndarray') -> 'np.ndarray'
    Given a face-vertex normal array, sums the normals at of each unique
symmetrical(self, pivot: 'float' = 0.0, axis: 'int' = 0, tolerance: 'float' = 1e-06, check_topology=True, check_normals=True) -> 'bool'
to_identity(self) -> 'None'
    sets the matrix to identity and transforms the points accordingly
to_prim(self, prim: 'Any') -> 'None'
    Streams data into a prim.
to_uvdata(self, name: 'str | None' = None, axis: 'str | Axis' = <Axis.Y: 'y'>, maintain_ratio=False, swap_uvs=False) -> 'UVData'
    creates an orthogonal projection of the mesh as a UVData object
triangulate(self, rules: 'Union[np.ndarray, TriangulateRules]', preserve_order: 'bool' = True) -> 'None'
    triangulates the mesh according to the given method.
union(self, *args, world_space=False)
    combines faces of given Mesh objects to self (this is not a boolean)
vertex_pairs_to_edges(self, verts: 'np.ndarray') -> 'np.ndarray'
    returns the edges for a given set of vertex pairs
vertices_to_edges(self, verts: 'np.ndarray') -> 'np.ndarray'
    returns the edges for a given set of vertices
vertices_to_faces(self, verts: 'np.ndarray') -> 'np.ndarray'
    returns the vertices for a given set of faces
```

#### `MeshList(DataList)`

All the operations on a read-write sequence.

```text
MeshList(iterable=None)
```

Methods:

```text
fix_symmetry(self, pivot: 'float' = 0.0, axis: 'int' = 0, side: 'float' = 1.0, combined: 'bool' = True, tolerance: 'Union[None, float]' = None) -> 'np.ndarray'
    attempts a symmetry fix on each element or combined.
classmethod load_fbx(cls, filename: 'str') -> "'MeshList'"
    loads all meshes from an fbx file into a MeshList
merge(self)
    applies merge to all elements
to_identity(self)
    sets all contained MeshData objects to identity
```

#### `RenderMethod(Enum)`

Create a collection of name/value pairs.

```text
RenderMethod(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)
```

#### `SampleMethod(Enum)`

Create a collection of name/value pairs.

```text
SampleMethod(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)
```

#### `TriangulateMethod(Enum)`

Create a collection of name/value pairs.

```text
TriangulateMethod(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)
```

#### `TriangulateRules`

Bundled triangulation rules for tri, quad, and n-gon faces.

```text
TriangulateRules(rules: 'np.ndarray', ngon_tris: 'dict') -> None
```

Fields:

```text
rules: np.ndarray
ngon_tris: dict
```

#### `UVData(MeshData)`

UV map topology treated as a Mesh object for resampling purposes

```text
UVData(indices: 'np.ndarray', counts: 'np.ndarray', points: 'np.ndarray', name: 'str' = 'map1', matrix: 'Optional[np.ndarray]' = ImmutableArray([[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]]), hole_faces: 'Optional[np.ndarray]' = None, hole_counts: 'Optional[np.ndarray]' = None, hole_indices: 'Optional[np.ndarray]' = None, normals: 'Optional[np.ndarray]' = None, normal_indices: 'Optional[np.ndarray]' = None) -> None
```

Fields:

```text
name: str = 'map1'
```

Properties:

```text
antialias                          
buffer                             returns the render buffer
has_holes                          returns True if the UVData has holes
pixel_counts                       
pixel_overlaps                     
pixel_ratios                       
renderer                           
resolution                         
```

Methods:

```text
buffer_from_file(self, fname: 'str') -> 'None'
    loads an image to the render buffer from a file
clear_buffer(self, color: 'Tuple[int, int, int]' = (0, 0, 0)) -> 'None'
    clears the render buffer
convolve(self, steps: 'int' = 1, kernel: 'np.ndarray | int' = 1) -> 'None'
    convolves (grows) the buffer over black pixels
staticmethod decode_rgb(rgb: 'Tuple[int, int, int] | np.ndarray') -> 'int | np.ndarray'
difference(self, *args, tolerance=1e-06)
    removes overlapping faces of given Mesh objects from self (this is not a boolean)
draw_edges(self, mask: 'UVData | np.ndarray | None' = None, indices: 'None | np.ndarray' = None, invert: 'bool' = False, color: 'Tuple[int, int, int] | np.ndarray | None' = (255, 255, 255), thickness: 'int' = 1, tolerance: 'float' = 0.0, clear_buffer: 'bool' = False) -> 'np.ndarray | None'
    draws edges
draw_faces(self, mask: 'UVData | np.ndarray | None' = None, indices: 'None | np.ndarray' = None, invert: 'bool' = False, color: 'Tuple[int, int, int] | np.ndarray | None' = (255, 255, 255), tolerance: 'float' = 0.0, clear_buffer: 'bool' = False) -> 'np.ndarray | None'
    draws faces
draw_mask(self, indices: 'None | np.ndarray' = None, invert: 'bool' = False, contour: 'bool' = False) -> 'None'
    draws a white mask
draw_points(self, mask: 'UVData | np.ndarray | None' = None, indices: 'None | np.ndarray' = None, invert: 'bool' = False, color: 'Tuple[int, int, int] | np.ndarray | None' = (255, 255, 255), radius: 'int' = 1, tolerance: 'float' = 0.0, clear_buffer: 'bool' = False) -> 'np.ndarray | None'
staticmethod encode_rgb(i: 'int | np.ndarray') -> 'Tuple[int, int, int] | np.ndarray'
erode(self, steps: 'int' = 1, kernel: 'np.ndarray | int' = 1) -> 'None'
    erodes (shrinks) the buffer's bordering black pixels
fill_holes(self, counts: 'np.ndarray') -> 'bool'
    fills empty faces with the expected face counts and default uvs
classmethod from_prim(cls, prim: 'Any') -> "List['UVData']"
    Construct a list of data objects from a mesh prim.
get_holes(self) -> 'np.ndarray'
    returns the missing face indices (aka holes)
get_minimum_resolution(self, pixels: 'int' = 3) -> 'np.ndarray'
    returns the minumum resolution to fit n pixels in a given bounding box
get_overlap_faces(self, uvdata: "'UVData'", indices: 'Union[None, np.ndarray]' = None, resolution: 'int' = 2048, tolerance: 'float' = 0.0)
    returns the self overlapping face indices over the given given UVData object
imshow(self) -> 'None'
    shows the image in a window
imwrite(self, fname: 'str') -> 'str'
    writes a png file
classmethod load_fbx(cls, filename: 'str', name: 'str | None' = None, channel: 'int' = 0) -> "'UVData'"
    returns a single UV channel from an fbx file (first mesh / channel 0 by default)
classmethod load_glb(cls, filename: 'str') -> "List['UVData']"
    a basic glb file reader to UVData
normalize(self, bbx_min=(0.0, 0.0), bbx_max=(1.0, 1.0), tolerance=1e-06)
    brings UV maps within frame range (default 0-1)
pack(self, resolution=1024, padding=2, rotations=4)
    Pack UV islands into [0,1]^2 using rasterized bitmap skyline algorithm.
patch_holes(self, uv_data: "'UVData'") -> 'bool'
    combines the given UVData object into this one
symmetrical(self, pivot: 'float' = 0.0, axis: 'int' = 0, tolerance: 'float' = 1e-06, check_topology=True)
    uv symmetry with pivot default to 0.5 in u
to_image(self) -> 'Image'
    creates a PIL Image object
to_prim(self, uv_set_id: 'int', prim: 'Any') -> 'None'
    Writes UV data into a mesh prim.
union(self, *args)
    combines faces of given Mesh objects to self (this is not a boolean)
```

#### `UVList(MeshList)`

All the operations on a read-write sequence.

```text
UVList(iterable=None)
```

Properties:

```text
has_holes                          returns True/False for each UVData if it has holes
```

Methods:

```text
defrag(self, keep_order: 'bool' = True) -> 'None'
    recombined fragmented UVData objects into complete UVData objects
fill_holes(self, counts: 'np.ndarray') -> 'np.ndarray'
    fills empty faces with the expected face counts and default uvs
fix_symmetry(self, pivot: 'float' = 0.5, axis: 'int' = 0, side: 'float' = 1.0, combined: 'bool' = True, tolerance: 'Union[None, float]' = None) -> 'np.ndarray'
    attempts a symmetry fix on each element or combined.
from_faces(self, *args, **kwarg) -> "'UVList'"
    returns a new UVList object from the given face indices
from_vertices(self, *args, **kwarg) -> "'UVList'"
    returns a new UVList object from the given vertex indices or rules
get_minimum_resolution(self)
    computes the minimum resolution for the entire list
classmethod load_fbx(cls, filename: 'str', name: 'str | None' = None) -> "'UVList'"
    loads all UV channels for one mesh from an fbx file (first mesh if name is None)
merge_seams(self)
    merges the UV seams
pack(self, resolution=1024, padding=2, rotations=4)
triangulate(self, rules: 'Union[np.ndarray, TriangulateRules]') -> 'None'
    applies the triangulation rules to all UVData in a UVList
```

Functions:

```text
load_fbx(file_path: 'str') -> 'list'
    loads fbx file and returns list of (MeshData, UVList) tuples
load_glb(file_path: 'str', scale_factor: 'float' = 100.0) -> 'list'
    loads glb file and returns list of (MeshData, UVList) tuples
load_obj(file_path: 'str') -> 'list'
    loads obj file and returns list of (MeshData, UVList) tuples
load_usd(file_path: 'str') -> 'list'
    loads usd file and returns list of (MeshData, UVList) tuples
save_obj(file_path: 'str', data: 'list') -> 'None'
    writes a valid obj file from a list of (MeshData, UVList) tuples
```

Constants:

```text
EPSILON = 1.1920929e-07
MESH_PRIM_TYPE = 'Mesh'
UV_ATTR_PREFIX = 'primvars:st'
```

## `cgmath.geometry.map`

painted maps and component subsets.

#### `GeomSubsetData(Data)`

Geometry subset data class.

```text
GeomSubsetData(name: 'str', indices: 'np.ndarray', component_type: 'str' = 'v', category: 'Optional[str]' = None) -> None
```

Fields:

```text
name: str
indices: np.ndarray
component_type: str = 'v'
category: Optional[str] = None
```

Methods:

```text
classmethod from_prim(cls, prim: 'Any') -> "'GeomSubsetData'"
    Constructs a data object from a prim.
classmethod from_skin_data(cls, skin_data: 'SkinData', mesh_data: 'MeshData | None' = None, **kwargs) -> 'GeomSubsetData'
    Converts a skin data to a GeomSubsetData object, assuming there is
split_to_array_by_clusters(self, mesh_data: 'MeshData | None' = None) -> 'list[GeomSubsetData]'
    Splits this GeomSubsetData object into a list of GeomSubsetData objects based on local vert clusters.
to_prim(self, prim: 'Any') -> 'None'
    Streams data into a prim.
to_skin_data(self, mesh_data: 'MeshData | None' = None, influence: 'str' = 'inf') -> 'SkinData'
    Converts this map object to a skin data object.
```

#### `MapData(Data)`

Maps/Painted Maps general data class.

```text
MapData(name: 'str', indices: 'np.ndarray', values: 'np.ndarray', default_value: 'Number' = 0, component_type: 'str' = 'v', categories: 'Optional[List[str]]' = None) -> None
```

Fields:

```text
name: str
indices: np.ndarray
values: np.ndarray
default_value: Number = 0
component_type: str = 'v'
categories: Optional[List[str]] = None
```

Methods:

```text
classmethod from_prim(cls, prim: 'Any') -> "'GeomSubsetData'"
    Constructs a data object from a prim.
classmethod from_skin_data(cls, skin_data: 'SkinData', mesh_data: 'MeshData | None' = None, **kwargs) -> 'MapData'
    Converts this skin data to a MapData object, assuming there is
to_dense_array(self, component_count: 'int') -> 'np.ndarray'
    Convert this object to a dense value array.
to_prim(self, prim: 'Any') -> 'None'
    Streams data into a prim.
to_skin_data(self, mesh_data: 'MeshData', influence: 'str' = 'inf') -> 'SkinData'
    Converts this map object to a skin data object.
```

Constants:

```text
FACE_MAP_PRIM_TYPE = 'FaceFloatMap'
GEOM_SUBSET_PRIM_TYPE = 'GeomSubset'
VERT_MAP_PRIM_TYPE = 'VertFloatMap'
```

## `cgmath.geometry.morph_target`

sparse point offsets.

#### `MorphData(Data)`

Morph target data class.

```text
MorphData(name: 'str', offsets: 'np.ndarray', indices: 'Optional[np.ndarray]' = None)
```

Fields:

```text
name: str
offsets: np.ndarray
indices: Optional[np.ndarray] = None
```

Properties:

```text
magnitudes                         returns the offset's magnitudes
max                                returns highest magnitude of the offsets
min                                returns lowest magnitude of the offsets
mse                                computes the mean square error of the offsets
size                               returns the count of datapoints
unit                               returns offsets as unit vectors
```

Methods:

```text
classmethod from_mesh_data(cls, base_mesh_data, target_mesh_data, target_name: 'str' = '', tolerance: 'float | None' = None) -> 'MorphData'
    Creates a MorphData object from two mesh data objects.
classmethod from_prim(cls, prim) -> 'MorphData'
    Constructs a data object from a prim.
is_zero(self, tolerance: 'float' = 0.0) -> 'bool'
    returns True if this morph target has no data or all offsets are zero
prune_offsets(self, tolerance: 'float | None' = None, neighbors: 'np.ndarray | None' = None) -> 'None'
    prunes offsets with magnidutes <= tolerance
sort(self)
    sorts indices.
to_dict(self) -> 'dict'
    returns the annotated data as a dict
to_prim(self, prim) -> 'None'
    Streams data into a prim.
```

#### `MorphList(DataList)`

All the operations on a read-write sequence.

```text
MorphList(iterable=None)
```

Properties:

```text
max                                returns highest magnitude for each shape
min                                returns lowest magnitude for each shape
mse                                computes the mean square error for each shape
size                               returns the datapoint count for each shape
```

Methods:

```text
prune_offsets(self, *args, **kwargs) -> 'None'
    prunes offsets with magidutes <= tolerance for each shape
remove_unused(self, tolerance: 'float' = 0.0) -> 'None'
    removes shapes that are unused (empty or all offsets zero within tolerance)
```

Constants:

```text
BLS_PRIM_TYPE = 'MorphTarget'
```

## `cgmath.geometry.skin_weights`

dense and compact skin weights.

> Optimized version of skin_weights.py with Numba-accelerated implementations.

#### `CompactSkinData(Data)`

Compact skin data class with optimized to_skin_data.

```text
CompactSkinData(max_influences: 'int', influence_indices: 'np.ndarray', weights: 'np.ndarray', influences: 'List[str]') -> None
```

Fields:

```text
max_influences: int
influence_indices: np.ndarray
weights: np.ndarray
influences: List[str]
```

Properties:

```text
valid                              checks whether the skin data is valid
```

Methods:

```text
classmethod from_prim(cls, prim: 'Any') -> "'CompactSkinData'"
    Constructs a data object from a prim.
round(self, n: 'int')
    round weights to n digits
to_prim(self, prim: 'Any') -> 'None'
    Streams data into a prim.
to_skin_data(self) -> "'SkinData'"
    Optimized: uses parallel Numba scatter operation.
```

#### `Patterns`

predefined positive/negative name search patterns

```text
Patterns()
```

#### `SkinData(Data)`

Skin data class with optimized operations.

```text
SkinData(weights: 'np.ndarray', influences: 'List[str]', name: 'Optional[str]' = None) -> None
```

Fields:

```text
weights: np.ndarray
influences: List[str]
name: Optional[str] = None
```

Properties:

```text
counts                             Optimized: uses parallel Numba kernel.
negative_patterns                  
patterns                           
positive_patterns                  
size                               returns the number of vertices
valid                              checks whether the skin data is valid
```

Methods:

```text
append(self, influence: 'str', weight: 'float' = 0.0)
    appends an influence to the end of the list
staticmethod conform(*objects: "'SkinData'") -> "'SkinData'"
    conforms SkinData objects so their influence lists are identical
extend(self, influences: 'List[str]', weights: 'Optional[List[float]]' = None)
    appends multiple influences to the end of the list
fix_symmetry(self, mesh_data, positive_patterns: 'Tuple[str] | None' = None, negative_patterns: 'Tuple[str] | None' = None, pivot: 'float' = 0.0, axis: 'int' = 0, side: 'float' = 1.0, tolerance: 'Union[None, float]' = None, force: 'bool' = False, max_influences: 'int' = -1) -> 'bool'
    Optimized: uses parallel Numba kernel for weight mirroring.
get_asymmetric_weights(self, mesh_data, positive_patterns: 'Tuple[str] | None' = None, negative_patterns: 'Tuple[str] | None' = None, pivot: 'float' = 0.0, axis: 'int' = 0, tolerance: 'Union[None, float]' = None) -> 'bool'
    Optimized: uses parallel Numba kernel for symmetry comparisons.
get_indices_over_max(self, count: 'int') -> 'np.ndarray'
    returns the indices of vertices with more than count influences
get_influences(self, queries: 'str | list[str] | tuple[str] | None' = None, exact: 'bool' = True) -> 'list[str]'
    uses fnmatch to find any influences that match the queries
get_max_influences(self) -> 'int'
    returns the maximum number of influences on vertices
get_symmetry_map(self, positive_patterns: 'Tuple[str] | None' = None, negative_patterns: 'Tuple[str] | None' = None) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'
    computes the symmetric influence indices according to given patterns
index(self, influence: 'str')
    returns the index of an influence name
inpaint(self, neighbors: 'np.ndarray', indices: 'np.ndarray', iterations: 'int | np.ndarray' = 1, receptions: 'float | np.ndarray' = 0.5, contributions: 'float | np.ndarray' = 1.0, tolerance: 'float' = 1e-07)
insert(self, index: 'int', influence: 'str', weight: 'float' = 0.0)
    inserts an influence at the specified index
is_equivalent(self, other: "'SkinData'") -> 'bool'
    checks whether the skin data is equivalent to another
classmethod load_fbx(cls, filename: 'str', name: 'str | None' = None) -> "'SkinData'"
    returns a single SkinData from an fbx file (first skinned mesh if name is None)
classmethod load_glb(cls, filename: 'str', name: 'str | None' = None) -> "'SkinData'"
    returns a single SkinData from a glb (first skinned primitive if name is None)
normalize(self, locked_influences=None, indices=None)
    Optimized: uses parallel Numba kernel when no locked influences.
prune(self, min_value: 'float' = 0.0, normalize: 'bool' = True, optimize: 'bool' = False)
    Optimized: uses parallel Numba kernel.
rank(self, influences: 'List[str]') -> 'np.ndarray'
    takes a list of influences and returns their ranked ids
remove(self, influence: 'str', optimize: 'bool' = False)
    removes an influence from the skin data
remove_unused(self, tolerance: 'float' = 0.0, normalize: 'bool' = True)
    removes unused influences (max > tolerance)
round(self, n: 'int')
    round weights to n digits
set_max_influences(self, count: 'int', optimize: 'bool' = False, sorting_method: 'str' = 'introselect')
    Optimized: uses parallel Numba kernel.
smooth(self, neighbors: 'np.ndarray', indices: 'np.ndarray | None' = None, iterations: 'int | np.ndarray' = 1, receptions: 'float | np.ndarray' = 0.5, contributions: 'float | np.ndarray' = 1.0)
sort(self)
    sorts influences alphabetically
subdivide(self, mesh_data, steps: 'int' = 1, keep_borders: 'bool' = False, keep_edges: 'bool' = False, keep_size: 'bool' = False) -> 'None'
    uses the Catmull-Clark algorithm to subdivide the skin weights
symmetrical(self, mesh_data, positive_patterns: 'Tuple[str] | None' = None, negative_patterns: 'Tuple[str] | None' = None, pivot: 'float' = 0.0, axis: 'int' = 0, tolerance: 'Union[None, float]' = None) -> 'bool'
    Optimized: uses parallel Numba kernel for symmetry comparisons.
to_compact_skin_data(self) -> 'CompactSkinData'
    Optimized: uses parallel Numba top-k gathering.
transfer_influences(self, src_influences: 'list', dst_influences: 'list', weighted: 'bool' = True) -> 'None'
    transfers weights from source influence(s) to destination influence(s)
```

#### `SkinList(DataList)`

All the operations on a read-write sequence.

```text
SkinList(iterable=None)
```

Methods:

```text
transfer_influences(self, src_influences: 'list', dst_influences: 'list', weighted: 'bool' = True) -> 'None'
    transfers weights from one influence to another
```

Functions:

```text
load_fbx(filename: 'str', bind_matrices: 'bool' = False) -> 'list'
    loads an fbx file and returns a list of ``(mesh name, SkinData)`` tuples
load_glb(filename: 'str', bind_matrices: 'bool' = False) -> 'list'
    loads a glb and returns one entry per primitive, ``None`` where unskinned
```

Constants:

```text
EPSILON = 1.1920929e-07
JOINT_NAMES_ATTR = 'skel:jointNames'
SKIN_PRIM_TYPE = 'SkinWeights'
```

## `cgmath.geometry.bspline`

B-spline curves.

#### `BSplineData(Data)`

BSpline data class.

```text
BSplineData(points: numpy.ndarray, degree: int, periodic: bool, uniform: bool = False, use_numba: bool = True, registered: bool = False, arc_length_samples: int = 1000) -> None
```

Fields:

```text
points: ndarray
degree: int
periodic: bool
uniform: bool = False
use_numba: bool = True
registered: bool = False
arc_length_samples: int = 1000
```

Properties:

```text
control_point_params               Monotonically increasing native parameter values for each control
count                              
cv                                 control vertices
geometry                           
kv                                 Unpadded knot vector.
length                             Returns the total arc length of the curve.
max_param                          
total_length                       Returns the total arc length of the curve.
```

Methods:

```text
basis(self, u, collapse=False, use_numba=None, _native=False)
    Return the B-spline basis functions at parameter u.
close(self)
    Closes an open curve to make it periodic.
compute(self, u)
    Computes points and tangents on curve at given u coordinate.
compute_fast(self, u)
    Compute points and tangents using the fastest available method (Numba).
get_length(self, fast: bool = True, samples: int = 100) -> float
    Approximates the length of the curve.
invalidate(self) -> None
    Invalidate all cached data.
open(self)
    Opens a periodic (closed) curve.
rebuild(self) -> None
    Rebuild all cached data from current control points.
rebuild_arc_length_table(self) -> None
    Rebuild cached spline and arc-length lookup table.
sample(self, obj, initial_samples=20, max_newton_iters=10, tolerance=1e-10)
    Projects points onto the curve using Newton-Raphson refinement.
smooth(self, n: int, polyorder: int = 3, lock_endpoints: bool = True, pad_endpoints: bool = True)
    smoothes the curve points via a savgol_filter
```

#### `SampleData(Data)`

A dataclass to hold bspline sample data

```text
SampleData(points: numpy.ndarray, tangents: numpy.ndarray, distances: numpy.ndarray, params: numpy.ndarray, basis: numpy.ndarray) -> None
```

Fields:

```text
points: ndarray
tangents: ndarray
distances: ndarray
params: ndarray
basis: ndarray
```

Methods:

```text
compute(self, values)
    computes weighted values from given data
copy(self)
    returns a deep copy of self
```

## `cgmath.geometry.bspline_patch`

tensor-product B-spline patches.

#### `BSplinePatchData(Data)`

B-spline tensor-product surface.

```text
BSplinePatchData(points: numpy.ndarray, degree_u: int, degree_v: int, periodic_u: bool, periodic_v: bool, uniform_u: bool = False, uniform_v: bool = False, use_numba: bool = True, registered_u: bool = False, registered_v: bool = False, arc_length_samples: int = 1000) -> None
```

Fields:

```text
points: ndarray
degree_u: int
degree_v: int
periodic_u: bool
periodic_v: bool
uniform_u: bool = False
uniform_v: bool = False
use_numba: bool = True
registered_u: bool = False
registered_v: bool = False
arc_length_samples: int = 1000
```

Properties:

```text
area                               
control_point_params               Per-direction parameter values for each control point row/column,
count_u                            
count_v                            
cv                                 Control vertices with periodic wrapping applied (cu, cv, dims).
geometry_u                         Parametric U index map (with periodic wrap).
geometry_v                         Parametric V index map (with periodic wrap).
kv_u                               Unpadded U knot vector.
kv_v                               Unpadded V knot vector.
length_u                           Fast approximation of total arc length along U (mid-V iso-curve).
length_v                           Fast approximation of total arc length along V (mid-U iso-curve).
max_param_u                        
max_param_v                        
total_length_u                     Total arc length along U.
total_length_v                     Total arc length along V.
```

Methods:

```text
basis(self, u, v, collapse=False, use_numba=None, _native=False)
    Tensor-product basis functions at (u, v).
close_u(self)
    Make U direction periodic.
close_v(self)
    Make V direction periodic.
compute(self, u, v, _native=False)
    Compute surface points and first partial derivatives.
compute_fast(self, u, v, _native=False)
    Always-numba positions + tangents. Mirrors ``BSplineData.compute_fast``.
evaluate(self, u, v, _native=False)
    Position-only fast path using the existing surface evaluation kernel.
get_area(self, samples: int = 50) -> float
    Numerical surface area via composite Simpson's rule on
get_length_u(self, fast: bool = True, samples: int = 100) -> float
    Approximate arc length along U using a mid-V iso-curve.
get_length_v(self, fast: bool = True, samples: int = 100) -> float
    Approximate arc length along V using a mid-U iso-curve.
invalidate(self) -> None
    Invalidate all cached data.
open_u(self)
    Make U direction non-periodic.
open_v(self)
    Make V direction non-periodic.
raycast(self, origins, directions=None, forward_only: bool = True, twosided: bool = True, initial_samples: int = 20, max_newton_iters: int = 20, tolerance: float = 1e-10, hit_eps: float = 0.0001)
    Ray-surface intersection on this B-spline patch.
rebuild(self) -> None
    Full cache rebuild.
rebuild_arc_length_table(self) -> None
    Rebuild cached spline, cv cache, and arc-length tables.
sample(self, obj, initial_samples=20, max_newton_iters=10, tolerance=1e-10)
    Project points onto the surface via Newton-Raphson refinement.
smooth(self, n_u: int, n_v: int, polyorder: int = 3)
    Separable Savgol smoothing along U then V with periodic wrapping.
```

#### `PatchRaycastData(Data)`

Ray-surface intersection data for a B-spline patch.

```text
PatchRaycastData(points: numpy.ndarray, tangents: numpy.ndarray, distances: numpy.ndarray, params: numpy.ndarray, basis: numpy.ndarray, occluded: numpy.ndarray, hit: numpy.ndarray) -> None
```

Fields:

```text
points: ndarray
tangents: ndarray
distances: ndarray
params: ndarray
basis: ndarray
occluded: ndarray
hit: ndarray
```

Properties:

```text
normals                            Surface normals via cross(dS/du, dS/dv), normalized.
```

Methods:

```text
compute(self, values)
    Compute weighted values from basis. values can be (nu, nv, d) or (nu*nv, d).
copy(self)
    returns a deep copy of self
```

#### `PatchSampleData(SampleData)`

Sample data for B-spline surface projections.

```text
PatchSampleData(points: numpy.ndarray, tangents: numpy.ndarray, distances: numpy.ndarray, params: numpy.ndarray, basis: numpy.ndarray) -> None
```

Properties:

```text
normals                            Surface normals via cross(dS/du, dS/dv), normalized.
```

Methods:

```text
compute(self, values)
    Compute weighted values from basis. values can be (nu, nv, d) or (nu*nv, d).
copy(self)
    returns a deep copy of self
```

## `cgmath.geometry._saddle_surface`

bilinear / PN-Quad sampling, raycast and integration.

#### `RaycastData(Data)`

A dataclass to hold ray-mesh intersection data

```text
RaycastData(projections: numpy.ndarray, distances: numpy.ndarray, weights: numpy.ndarray, indices: numpy.ndarray, normals: numpy.ndarray, occluded: numpy.ndarray, uvs: numpy.ndarray, geometry: numpy.ndarray, hit: numpy.ndarray) -> None
```

Fields:

```text
projections: ndarray
distances: ndarray
weights: ndarray
indices: ndarray
normals: ndarray
occluded: ndarray
uvs: ndarray
geometry: ndarray
hit: ndarray
```

Methods:

```text
compute(self, values)
    computes weighted values from given data
```

#### `SampleData(Data)`

A dataclass to hold saddle surface sample data

```text
SampleData(projections: numpy.ndarray, distances: numpy.ndarray, weights: numpy.ndarray, indices: numpy.ndarray, normals: numpy.ndarray, occluded: numpy.ndarray, uvs: numpy.ndarray, geometry: numpy.ndarray) -> None
```

Fields:

```text
projections: ndarray
distances: ndarray
weights: ndarray
indices: ndarray
normals: ndarray
occluded: ndarray
uvs: ndarray
geometry: ndarray
```

Methods:

```text
compute(self, values)
    computes weighted values from given data
remap(self, src_mesh, dst_mesh, dst_uv)
    saddle surface sampler where closest face index is returned
```

Functions:

```text
integrate(points, geometry, samples=10, method='gauss', tolerance=0.0001)
    integrates the given topology and returns the surface area approximation
raycast(points, face_points, face_geometry, origins, directions, normals=None, surface_normals=None, forward_only=True, twosided=True, bvh=None)
    Ray-mesh intersection on bilinear or bicubic Bezier patches.
sample(points, face_points, face_geometry, normals=None, surface_normals=None, iteration_count=100, iteration_tolerance=1e-08, uv_border_tolerance=1e-05, perfect_match_tolerance=1e-06)
    preprocess data before going into the saddle surface algorithm
```

## `cgmath.geometry._cdt`

constrained Delaunay triangulation.

> Constrained Delaunay Triangulation (CDT) for arbitrary n-gon faces.

#### `TriMesh`

Half-edge-style triangle mesh for CDT operations.

```text
TriMesh(pts)
```

Methods:

```text
add_tri(self, a, b, c)
edges_as_array(self)
    Return all current edges as an ``(M, 2)`` int64 numpy array.
flip_edge(self, a, b)
has_edge(self, a, b)
opposite_vertex(self, tid, edge)
remove_tri(self, tid)
```

Functions:

```text
enforce_constraints(mesh, constrained_edges)
    Insert constrained edges into the mesh via Sloan's algorithm.
flatten_3d_to_2d(points_3d)
    Project 3D n-gon vertices onto their best-fit 2D plane.
get_boundary_edges(boundary)
    Return the set of canonical edge keys for a closed vertex loop.
maya_tiebreak(mesh)
    Swap near-cocircular edges to prefer the longer diagonal.
remove_exterior_and_holes(mesh, outer_boundary, hole_boundaries, points)
    Remove triangles outside outer boundary and inside any holes.
restore_delaunay(mesh)
    Restore the Delaunay property by flipping non-constrained edges.
triangulate_ngon(points, face_vertices, hole_boundaries=None)
    Triangulate an n-gon face using Constrained Delaunay Triangulation.
```

## `cgmath.geometry.sdf`

signed distance fields and dual marching cubes.

> Dual Marching Cubes (DMC) implementation for isosurface extraction.

#### `CSGNode`

Internal node holding a primitive and its CSG operation.

```text
CSGNode(primitive: Union[cgmath.geometry.sdf.SDFSphere, cgmath.geometry.sdf.SDFBox, cgmath.geometry.sdf.SDFCylinder], operation: cgmath.geometry.sdf.CSGOp, smoothing: float = 0.0) -> None
```

#### `CSGOp(str, Enum)`

CSG operation types for combining SDF primitives.

```text
CSGOp(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)
```

#### `DMCField`

A field that combines SDF primitives via CSG operations.

```text
DMCField(resolution: Union[int, Tuple[int, int, int]] = (64, 64, 64), position: Union[List[float], numpy.ndarray, NoneType] = None, iso_value: float = 0.0, name: str = 'dmc_mesh') -> None
```

Properties:

```text
iso_value                          Extraction threshold for the isosurface.
mesh_data                          The extracted mesh.
name                               Mesh name.
position                           World-space offset of the field center.
primitives                         All registered primitives.
resolution                         Subdivisions per world-space unit ``(rx, ry, rz)``.
```

Methods:

```text
add(self, primitive: Union[cgmath.geometry.sdf.SDFSphere, cgmath.geometry.sdf.SDFBox, cgmath.geometry.sdf.SDFCylinder], smoothing: Optional[float] = None) -> 'DMCField'
    Add a primitive via union (or smooth union if smoothing > 0).
intersect(self, primitive: Union[cgmath.geometry.sdf.SDFSphere, cgmath.geometry.sdf.SDFBox, cgmath.geometry.sdf.SDFCylinder]) -> 'DMCField'
    Intersect with a primitive.
remove(self, primitive: Union[cgmath.geometry.sdf.SDFSphere, cgmath.geometry.sdf.SDFBox, cgmath.geometry.sdf.SDFCylinder]) -> 'DMCField'
    Remove a primitive from the field.
subtract(self, primitive: Union[cgmath.geometry.sdf.SDFSphere, cgmath.geometry.sdf.SDFBox, cgmath.geometry.sdf.SDFCylinder]) -> 'DMCField'
    Subtract a primitive via difference.
to_obj(self, filename: str) -> None
    Export the mesh to an OBJ file.
```

#### `SDFBox(TransformData)`

Axis-aligned box SDF primitive inheriting transform properties from TransformData.

```text
SDFBox(half_extents: Union[List[float], numpy.ndarray, NoneType] = None, translate: Union[List[float], numpy.ndarray, NoneType] = None, rotate: Union[List[float], numpy.ndarray, NoneType] = None, scale: Union[List[float], numpy.ndarray, NoneType] = None, name: str = 'box') -> None
```

Properties:

```text
half_extents                       Half-extents [hx, hy, hz] along each axis.
```

Methods:

```text
bounding_box(self) -> Tuple[numpy.ndarray, numpy.ndarray]
    Box AABB: abs(R) @ (half_extents * scale) offset by translate.
evaluate(self, X: numpy.ndarray, Y: numpy.ndarray, Z: numpy.ndarray) -> numpy.ndarray
    Evaluate the SDF on pre-transformed (local) coordinates.
sample(self, X: numpy.ndarray, Y: numpy.ndarray, Z: numpy.ndarray) -> numpy.ndarray
    Evaluate the SDF on world-space coordinates.
```

#### `SDFCylinder(TransformData)`

Capped cylinder SDF primitive inheriting transform properties from TransformData.

```text
SDFCylinder(radius: float = 0.5, height: float = 1.0, axis: int = 1, translate: Union[List[float], numpy.ndarray, NoneType] = None, rotate: Union[List[float], numpy.ndarray, NoneType] = None, scale: Union[List[float], numpy.ndarray, NoneType] = None, name: str = 'cylinder') -> None
```

Properties:

```text
axis                               Axis the cylinder is aligned with (0=X, 1=Y, 2=Z).
height                             Total height of the cylinder.
radius                             Cylinder radius.
```

Methods:

```text
bounding_box(self) -> Tuple[numpy.ndarray, numpy.ndarray]
    Cylinder AABB: compute local extent, rotate, take AABB of rotated box.
evaluate(self, X: numpy.ndarray, Y: numpy.ndarray, Z: numpy.ndarray) -> numpy.ndarray
    Evaluate the SDF on pre-transformed (local) coordinates.
sample(self, X: numpy.ndarray, Y: numpy.ndarray, Z: numpy.ndarray) -> numpy.ndarray
    Evaluate the SDF on world-space coordinates.
```

#### `SDFSphere(TransformData)`

Sphere SDF primitive inheriting transform properties from TransformData.

```text
SDFSphere(radius: float = 1.0, translate: Union[List[float], numpy.ndarray, NoneType] = None, rotate: Union[List[float], numpy.ndarray, NoneType] = None, scale: Union[List[float], numpy.ndarray, NoneType] = None, name: str = 'sphere') -> None
```

Properties:

```text
radius                             Sphere radius.
```

Methods:

```text
bounding_box(self) -> Tuple[numpy.ndarray, numpy.ndarray]
    Sphere AABB: abs(R) @ (radius * scale) offset by translate.
evaluate(self, X: numpy.ndarray, Y: numpy.ndarray, Z: numpy.ndarray) -> numpy.ndarray
    Evaluate the SDF on pre-transformed (local) coordinates.
sample(self, X: numpy.ndarray, Y: numpy.ndarray, Z: numpy.ndarray) -> numpy.ndarray
    Evaluate the SDF on world-space coordinates.
```

Functions:

```text
box_sdf(shape: Tuple[int, int, int], center: Optional[numpy.ndarray] = None, half_extents: Optional[numpy.ndarray] = None, bounds: Optional[Tuple[numpy.ndarray, numpy.ndarray]] = None) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    Generate a box signed distance field for testing.
cylinder_sdf(shape: Tuple[int, int, int], center: Optional[numpy.ndarray] = None, radius: float = 0.5, height: float = 1.0, axis: int = 1, bounds: Optional[Tuple[numpy.ndarray, numpy.ndarray]] = None) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    Generate a capped cylinder signed distance field for testing.
dual_marching_cubes(scalar_field: numpy.ndarray, iso_value: float = 0.0, grid_origin: Optional[numpy.ndarray] = None, grid_spacing: Optional[numpy.ndarray] = None, name: Optional[str] = None) -> cgmath.geometry.mesh.MeshData
    Generate a mesh from a scalar field using Dual Marching Cubes.
eval_box(X: numpy.ndarray, Y: numpy.ndarray, Z: numpy.ndarray, half_extents: numpy.ndarray) -> numpy.ndarray
    Evaluate an axis-aligned box SDF centered at the origin.
eval_cylinder(X: numpy.ndarray, Y: numpy.ndarray, Z: numpy.ndarray, radius: float = 0.5, height: float = 1.0, axis: int = 1) -> numpy.ndarray
    Evaluate a capped cylinder SDF centered at the origin.
eval_sphere(X: numpy.ndarray, Y: numpy.ndarray, Z: numpy.ndarray, radius: float = 1.0) -> numpy.ndarray
    Evaluate a sphere SDF centered at the origin.
make_grid(shape: Tuple[int, int, int], bounds: Tuple[numpy.ndarray, numpy.ndarray]) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    Create a regular 3D sample grid.
sdf_difference(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray
    Boolean difference (A \ B).  Subtracts B from A.
sdf_intersection(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray
    Boolean intersection (A intersect B).  Keeps only the overlap.
sdf_smooth_difference(a: numpy.ndarray, b: numpy.ndarray, k: float = 0.1) -> numpy.ndarray
    Smooth (filleted) difference -- subtracts B from A with blended edge.
sdf_smooth_intersection(a: numpy.ndarray, b: numpy.ndarray, k: float = 0.1) -> numpy.ndarray
    Smooth (filleted) intersection.
sdf_smooth_union(a: numpy.ndarray, b: numpy.ndarray, k: float = 0.1) -> numpy.ndarray
    Smooth (filleted) union.
sdf_union(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray
    Boolean union (A union B).  Keeps both shapes.
sphere_sdf(shape: Tuple[int, int, int], center: Optional[numpy.ndarray] = None, radius: float = 1.0, bounds: Optional[Tuple[numpy.ndarray, numpy.ndarray]] = None) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    Generate a sphere signed distance field for testing.
transform_points(X: numpy.ndarray, Y: numpy.ndarray, Z: numpy.ndarray, *, translation: Optional[numpy.ndarray] = None, rotation: Optional[numpy.ndarray] = None, scale: Union[float, numpy.ndarray, NoneType] = None) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    Map world-space grid points into local object space for SDF evaluation.
```

Constants:

```text
CORNER_OFFSETS = array([[0, 0, 0],
       [1, 0, 0],
       [1, 1, 0],
   ...
EDGE_AXIS = array([0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2])
EDGE_TABLE = array([   0,  265,  515,  778, 1030, 1295, 1541, 1804, 20...
EDGE_VERTICES = array([[0, 1],
       [1, 2],
       [2, 3],
       [3, 0...
FACE_CUBE_OFFSETS_X = array([[ 0,  0,  0],
       [ 0, -1,  0],
       [ 0, -1,...
FACE_CUBE_OFFSETS_Y = array([[ 0,  0,  0],
       [-1,  0,  0],
       [-1,  0,...
FACE_CUBE_OFFSETS_Z = array([[ 0,  0,  0],
       [-1,  0,  0],
       [-1, -1,...
```

## `cgmath.geometry.pack`

UV island packing.

> UV island packing via rasterized bitmap skyline algorithm.

Functions:

```text
pack_islands(uv_data, resolution=1024, padding=2, rotations=4)
    Pack UV islands into [0,1]^2 using rasterized bitmap skyline algorithm.
```

## `cgmath.geometry.resample`

topology transfer of per-vertex data.

> A resampler class interfacing mesh-dependent data types (morph targets and skin weights).

#### `MeshDataResampler`

Resampler/transfer class for mesh-dependent data types.

```text
MeshDataResampler(src_mesh: 'MeshData', dst_mesh: 'MeshData', src_uv: 'UVData | None' = None, dst_uv: 'UVData | None' = None) -> 'None'
```

Properties:

```text
dst_mesh                           Returns the destination mesh data.
dst_uv                             Returns the destination uv data.
spatial_sample_data                Cache and return a sample data object in world space (bilinear).
src_mesh                           Returns the source mesh data.
src_uv                             Returns the source uv data.
uv_sample_data                     Cache and return a sample data object in UV space.
```

Methods:

```text
get_sample_data(self, mode: 'ResampleMode', method: 'SampleMethod | str' = <SampleMethod.BILINEAR: 'bilinear'>) -> 'SampleData'
    Returns a cached sample data object based on mode and method.
resample_map(self, map_data: 'MapData | GeomSubsetData', method: 'SampleMethod | str' = <SampleMethod.BILINEAR: 'bilinear'>, **kwargs) -> 'MapData | GeomSubsetData'
    Resamples a map data.
resample_mesh(self, mesh_data: 'MeshData | None' = None, mode: 'ResampleMode' = <ResampleMode.SPATIAL: <class 'cgmath.geometry.resample.SpatialSkinTransferOptions'>>, method: 'SampleMethod | str' = <SampleMethod.BILINEAR: 'bilinear'>, maintain_offset: 'bool' = False, orient_offset: 'bool' = True)
    Resamples mesh_data
resample_morph_target(self, target_data: 'MorphData | MeshData', mode: 'ResampleMode' = <ResampleMode.SPATIAL: <class 'cgmath.geometry.resample.SpatialSkinTransferOptions'>>, method: 'SampleMethod | str' = <SampleMethod.BILINEAR: 'bilinear'>, tolerance: 'float | None' = None, use_neighbors: 'bool' = True) -> 'MorphData | MeshData'
    Resamples a morph target data.
resample_skin_weights(self, skin_data: 'SkinData | CompactSkinData', mode: 'ResampleMode' = <ResampleMode.SPATIAL: <class 'cgmath.geometry.resample.SpatialSkinTransferOptions'>>, method: 'SampleMethod | str' = <SampleMethod.BILINEAR: 'bilinear'>, options: 'Any' = None) -> 'SkinData | CompactSkinData'
    Resamples a skin data.
```

#### `ResampleMode(Enum)`

Data resample modes.

```text
ResampleMode(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)
```

#### `SpatialSkinTransferOptions`

SpatialSkinTransferOptions(max_influences: 'int' = 4)

```text
SpatialSkinTransferOptions(max_influences: 'int' = 4) -> None
```

Fields:

```text
max_influences: int = 4
```

#### `UVSkinTransferOptions`

UVSkinTransferOptions(max_influences: 'int' = 4)

```text
UVSkinTransferOptions(max_influences: 'int' = 4) -> None
```

Fields:

```text
max_influences: int = 4
```

## `cgmath.geometry.robust_skinweights_transfer_bilinear`

robust weight inpainting, quad-native.

> Robust skin weights transfer via weight inpainting, on bilinear patches.

#### `RobustBilinearSkinTransferOptions`

User set parameters for controlling the behavior of the robust bilinear skin transfer algorithm.

```text
RobustBilinearSkinTransferOptions(max_influences: 'int' = 4, search_radius: 'float' = 0.05, normal_threshold: 'float' = 30, inpaint_iterations: 'int' = 10, smooth_iterations: 'int' = 10, smooth_strength: 'float' = 0.1) -> None
```

Fields:

```text
max_influences: int = 4
search_radius: float = 0.05
normal_threshold: float = 30
inpaint_iterations: int = 10
smooth_iterations: int = 10
smooth_strength: float = 0.1
```

Functions:

```text
find_matches_closest_surface(sample_data: 'SampleData', dst_mesh: 'MeshData', src_skin_weights: 'np.ndarray', search_radius: 'float', normal_threshold_deg: 'float') -> 'tuple[np.ndarray, np.ndarray]'
    For each vertex on the target mesh find a match on the source mesh.
robust_skinweights_transfer_bilinear(src_mesh: 'MeshData', src_skin_data: 'SkinData', dst_mesh: 'MeshData', transfer_options: 'RobustBilinearSkinTransferOptions | None' = None, sample_data: 'SampleData | None' = None) -> 'np.ndarray'
    Transfers skin weights from a source mesh onto a target mesh.
```

## `cgmath.geometry.surface_plotting`

surface-space control retargeting.

Functions:

```text
batch_retarget_init(function_data: dict, rig_args: dict, mesh_data: dict, uvn_scale: Optional[List[float]] = None, refactor_alignment: bool = False) -> dict
    Initializes the retargeting process for a set of rigs, adjusting their function data to a new space.
bspline_weigh_transformations(control_data: dict, bspline_map: dict, Mesh: Optional[object] = None, UVList: Optional[list] = None, uv_map_index: int = 0, normal_scale: float = 1.0, return_matrices: bool = False) -> dict
    Computes the weighted transformation for objects based on B-spline mapping and optional surface constraints.
build_normal_matrix(input_point: numpy.ndarray, normal_scale: float = 1.0) -> numpy.ndarray
control_array_type_conversion(array: list, to_transformation_matrix: bool = False) -> numpy.ndarray
    Converts an array of points or matrices into a different format based on the specified flag.
control_sort(lst: List[str]) -> List[str]
    Sorts a list of control names by separating and ordering them based on specific suffixes.
convert_dict_vals_arrays(dictionary: dict, to_numpy_array: bool = False) -> dict
    Recursively iterate over a dictionary and convert any numpy arrays to lists.
convert_points_to_surfacespace_matrices(input_points: list, Mesh: object, UVList: list, uv_map_index: int = 0) -> numpy.ndarray
    Converts input points to transformation matrices in surface space.
extract_scale_matrix(transformation_matrix: numpy.ndarray) -> numpy.ndarray
    Extracts the scale components from a 4x4 transformation matrix.
find_key(dictionary: dict, key: str) -> list
    Find all instances of a specific key in a dictionary.
get_bspline_map_init(control_data: Tuple[List[str], numpy.ndarray], driven_data: Optional[Tuple[List[str], numpy.ndarray]] = None, Mesh: Optional[object] = None, UVList: Optional[List[object]] = None, uv_map_index: int = 0, uvn_scale: Optional[List[float]] = None, refactor_inputs: Optional[Tuple[List[str], numpy.ndarray]] = None, refactor_alignment: bool = False) -> Dict[str, Union[List, Dict]]
    Initializes a B-spline mapping for control and driven data, potentially refactoring the control matrices.
get_control_arrays(controls_list: List[str]) -> Dict[str, List[str]]
    Organizes a list of control names into a dictionary based on their root names.
get_function_pose(rig_function_data: dict, mesh_data: dict, pose: str, relative: bool = False) -> dict
    Retrieves the pose data for a given rig and pose name, transforming it into the appropriate space.
get_points_from_surface(Mesh: object, UVList: list, input_points: list, uv_map_index: int = 0, as_3d: bool = False) -> numpy.ndarray
    Retrieves points from a surface based on input points and UV mapping.
get_roots(controls_list: List[str]) -> List[str]
    Extracts and returns the unique root names from a list of control names.
refactor_control_object_matrices(control_matrices: numpy.ndarray, refactor_data: dict, refactor_alignment: bool = False) -> dict
    Refactors control object matrices based on provided refactor data.
refactor_control_object_matrices_init(control_matrices: numpy.ndarray, uvn_scale: list = None, refactor_inputs: list = None, refactor_alignment: bool = False) -> dict
    Initializes the refactoring of control object matrices based on provided inputs.
reset_scale(transformation_matrix: numpy.ndarray) -> numpy.ndarray
    Resets the scale component of a transformation matrix to identity.
retarget_function_data(function_data, bslpine_map_data, refactor_alignment=False)
set_value_by_path(dictionary: dict, path: list, value) -> None
    Set a value in a dictionary by a given path.
surface_points_to_transformation_matrices(positions: numpy.ndarray, u_vecs: numpy.ndarray, v_vecs: numpy.ndarray) -> dict
    Converts surface points and their corresponding U and V vectors into transformation matrices.
transforms_from_surface_coordinates(Mesh: object, UVList: list, points: list, uv_map_index: int = 0, z_scale: float = 1.0) -> dict
    Transforms points from surface coordinates to 3D space using the given mesh and UV list.
transforms_to_surface_space(Mesh: object, UVList: list, matrices: numpy.ndarray, uv_map_index: int = 0, as_matrices: bool = False) -> numpy.ndarray
    Converts transformation matrices to surface space coordinates.
u_vector_to_rotation_matrix(input_vector: numpy.ndarray) -> numpy.ndarray
    Converts a given U vector into a rotation matrix.
```

## `cgmath.geometry.deform`

the five deformers.

> Move all data types into the same namespace

Re-exported classes:

```text
DeformMethod
DeltaMushData
FFDData
PatchRelaxData
SkinDeformData
WrapData
```

## `cgmath.geometry.utils`

the numba kernel floor, as re-exported from `utils/main.py`.

Functions:

```text
assign_cells(points: 'np.ndarray', lattice: 'np.ndarray', max_hops: 'int' = 4) -> 'tuple[np.ndarray, np.ndarray]'
    For each point find its containing cell and parametric (u, v, w).
average_points(points, geometry)
    Optimized: uses parallel Numba kernel for averaging.
balance_center_weights(weights: 'np.ndarray', positive: 'np.ndarray', negative: 'np.ndarray', center: 'np.ndarray', max_inf: 'int')
    balances the weights of a center vertices up to max_inf
batch_procrustes_rotations(H: 'np.ndarray', max_iter: 'int' = 20) -> 'np.ndarray'
    Per-cluster polar-decomposition rotations from covariance matrices.
bernstein_basis_1d(t: 'np.ndarray', degree: 'int') -> 'np.ndarray'
    Bernstein polynomial basis weights.
bernstein_eval(cells: 'np.ndarray', uvw: 'np.ndarray', delta: 'np.ndarray', local_influence: 'tuple[int, int, int]', divisions: 'np.ndarray') -> 'np.ndarray'
    Bernstein polynomial FFD evaluation (Sederberg & Parry formulation).
bezier_raycast(points, geometry, normals, origins, directions, twosided=True, bvh=None)
    Forward-only ray-mesh intersection on bicubic Bezier patches (PN Quads).
bilinear_integrate(points, geometry, samples=10, method='gauss', tolerance=0.0001)
    returns bilinear quad surface areas
bilinear_raycast(points, geometry, origins, directions, twosided=True, bvh=None)
    Forward-only ray-mesh intersection on bilinear patches.
bilinear_sample(p, points, geometry, centroids, radiuses, centroid_distance_tolerance, iteration_count=100, iteration_tolerance=1e-08, uv_border_tolerance=1e-05, normals=None)
    returns a bilinear surface sample
bilinear_vectors(points, geometry, uv)
    returns bilinear quad surface vectors at given uvs
blend_deltas(base_points: 'np.ndarray', deltamush_points: 'np.ndarray', weight: 'float') -> 'np.ndarray'
    Linear blend ``base + weight * (deltamush - base)`` per vertex.
blur(weights: 'np.ndarray', neighbors: 'np.ndarray', iterations: 'int | np.ndarray' = 9, receptions: 'float | np.ndarray' = 0.5, contributions: 'float | np.ndarray' = 1.0)
    a blur algorithm with default values tuned to match Maya's delta mush
build_lattice_topology(lx: 'int', ly: 'int', lz: 'int') -> 'tuple[np.ndarray, np.ndarray]'
    Build quad face indices/counts for a structured (lx, ly, lz) grid.
build_tri_expand_map(counts, rules)
    Map original face-vertex stream positions to triangulated positions.
build_uv_to_normal_map(uv_indices, normal_indices, num_uv_verts)
    Map each UV vertex to a normal slot via face-vertex stream alignment.
build_vertex_rings(indices: 'np.ndarray', counts: 'np.ndarray', point_count: 'int') -> 'tuple[np.ndarray, np.ndarray, np.ndarray]'
    Build winding-ordered 1-ring adjacency for every vertex.
compare_normals(normal_a, normal_b)
    Optimized: uses parallel Numba kernel for angle computation.
composite_sprite(sprite: 'np.ndarray', buffer: 'np.ndarray', mask: 'np.ndarray | None' = None, offset: 'np.ndarray | tuple[int, int]' = (0, 0), color: 'np.ndarray | tuple[int, int, int]' = (255, 255, 255))
    composites a sprite to a buffer with offset and computes the overlap to a mask
composite_sprite_aa(sprite: 'np.ndarray', buffer: 'np.ndarray')
    composites an antialiased sprite to anothert sprite buffer
compute_basis(u, kv, c, d)
    returns bilinear quad surface areas
compute_centroids(points, geometry, return_radiuses=False)
    Optimized: uses parallel Numba kernel for centroid computation.
compute_decal_maps(points: 'np.ndarray', ring: 'np.ndarray', valence: 'np.ndarray', is_boundary: 'np.ndarray', out: 'np.ndarray | None' = None) -> 'np.ndarray'
    Flatten each vertex's edge stencil into geodesic polar coordinates.
compute_e2v_e2f(f2v, counts)
    builds edges_to_vertices and edges_to_faces components
compute_neighbor_distances(neighbors: 'np.ndarray', points: 'np.ndarray') -> 'np.ndarray'
    Compute Euclidean distances for each edge in the neighbor matrix.
compute_neighbors(i2x, x2i, n: 'int' = 0)
    builds vertex neighbors given v2x and x2v matrices
compute_samples(values, weights, geometry)
compute_span_weights(decals: 'np.ndarray', valence: 'np.ndarray') -> 'np.ndarray'
    Span-aware edge weights from a decal map.
compute_topological_neighborhood(connectivity, num_hops=None, indices=None, distances=None, max_distance=None, distance_format='sparse')
    Compute the topological neighborhood of vertices within num_hops. (Dijkstra's algorithm)
compute_v2x(f2v, shape)
    general purpose vertex_to_x builder
ddm_precompute(smooth_points: 'np.ndarray', neighbors: 'np.ndarray') -> 'tuple[np.ndarray, np.ndarray]'
    Precompute weighted neighbour offsets for the DDM apply step.
decode_local_deltas(smooth_points: 'np.ndarray', neighbors: 'np.ndarray', deltas: 'np.ndarray') -> 'np.ndarray'
    Reconstruct world positions by re-applying stored local deltas.
decode_local_deltas_tbn(smooth_points: 'np.ndarray', first_nbrs: 'np.ndarray', deltas: 'np.ndarray') -> 'np.ndarray'
    Reconstruct world positions using first-face TBN frames.
decode_world_deltas_ddm(smooth_def: 'np.ndarray', neighbors: 'np.ndarray', rest_offsets: 'np.ndarray', rest_weights: 'np.ndarray', world_deltas: 'np.ndarray') -> 'np.ndarray'
    Apply Direct-Delta-Mush per-vertex rotation to world-space deltas.
decode_world_deltas_procrustes(smooth_rest: 'np.ndarray', smooth_def: 'np.ndarray', neighbors: 'np.ndarray', world_deltas: 'np.ndarray') -> 'np.ndarray'
    Apply per-vertex Procrustes (SVD) rotation to world-space deltas.
encode_local_deltas(rest_points: 'np.ndarray', smooth_points: 'np.ndarray', neighbors: 'np.ndarray') -> 'np.ndarray'
    Encode each rest vertex displacement in its smoothed local tangent frame.
encode_local_deltas_tbn(rest_points: 'np.ndarray', smooth_points: 'np.ndarray', first_nbrs: 'np.ndarray') -> 'np.ndarray'
    Encode rest displacement in each vertex's first-face TBN frame.
get_cell_corners(lattice: 'np.ndarray', cells: 'np.ndarray') -> 'np.ndarray'
    Gather 8 corners of each hex cell from a structured lattice.
indices_replace(matrix, from_indices, to_indices, collapse=False)
    Optimized: uses parallel Numba kernel with binary search.
inpaint(indices: 'np.ndarray', weights: 'np.ndarray', neighbors: 'np.ndarray', iterations: 'int | np.ndarray' = 9, receptions: 'float | np.ndarray' = 0.5, contributions: 'float | np.ndarray' = 1.0)
    a blur algorithm with default values tuned to match Maya's delta mush
inverse_trilinear(points: 'np.ndarray', corners: 'np.ndarray', uvw_init: 'np.ndarray | None' = None, max_iter: 'int' = 20, tol: 'float' = 1e-10) -> 'np.ndarray'
    Newton-Raphson solve for parametric coords.
matrix_index_lookup(matrix, indices)
matrix_row_combine(matrix, from_rows, to_rows)
    creates a new resized matrix where elements of `from_rows`
matrix_row_overlaps(matrix, exact=True, sort_indices=False)
    Optimized: uses Numba hash-based row comparison.
matrix_to_stream(matrix)
    converts matrix to index stream, skips over -1 pads
minimize_neighbor_distances(neighborhood: 'np.ndarray', neighborhood_dists: 'np.ndarray', num_vertices: 'int') -> 'np.ndarray'
    Find the shortest distance to each unique vertex index across all rows.
patch_relax(points: 'np.ndarray', rest_points: 'np.ndarray', ring: 'np.ndarray', valence: 'np.ndarray', is_boundary: 'np.ndarray', weights: 'np.ndarray', rest_vectors: 'np.ndarray', rest_decals: 'np.ndarray', iterations: 'int' = 30, alpha: 'float' = 1.0, step_size: 'float' = 0.5, surface_blend: 'float' = 0.0, step_scale: 'np.ndarray | None' = None, polar_iters: 'int' = 12) -> 'np.ndarray'
    Run patch-based surface relaxation on a deformed pose.
point_row_overlaps(matrix, tolerance=1e-06)
    Optimized: uses Numba for parallel pair processing.
pxr()
    Returns the pxr module (USD) or raises an ImportError
quad_match_greedy(score, fa, fb, n_faces)
    Greedy maximum-weight matching on the triangle dual graph.
rebuild_indices(new_indices, counts, cumsum, f2v, f2e, e2v, face_offset, edge_offset)
    rebuilds the topology inplace
remap_indices(vert_indices, uv_indices, distances)
    returns closest unique corresponding streams of indices
shared_edges_test(edges)
    a numba powered test to see if edges are repeated
split_at_hard_edges(indices, counts, e2v, e2f, hard_edges)
    Build face-varying normal slot indices via union-find at smooth edges.
stream_to_matrix(values, counts)
    Optimized: uses parallel Numba kernel for scatter operation.
subdivide_catmull_clark(points, indices, counts, f2v, f2e, e2v, e2f, v2f, v2e, keep_borders=False, keep_edges=False, border_verts=None)
    Perform one step of Catmull-Clark subdivision using optimized Numba kernels.
trilinear(uvw: 'np.ndarray', corners: 'np.ndarray') -> 'np.ndarray'
    Forward trilinear interpolation.
trilinear_jacobian(uvw: 'np.ndarray', corners: 'np.ndarray') -> 'np.ndarray'
    Jacobian of the trilinear mapping dP/d(uvw).
vector_angle_difference(vector_a, vector_b)
    Optimized: uses parallel Numba kernel for angle computation.
```

## `cgmath.render`

software raytracer, scene graph, cameras, textures, frames.

> Software raytracer rendering pipeline for ``MeshData`` / ``UVData`` pairs.

Re-exported classes:

```text
Camera
Frame
Light
Object
Scene
```

Functions:

```text
load_texture(texture: 'Union[str, np.ndarray]') -> 'np.ndarray'
    Returns a contiguous ``(H, W, 3)`` float32 array in [0, 1].
look_at(eye: 'Tuple[float, float, float]', target: 'Tuple[float, float, float]', up: 'Tuple[float, float, float]' = (0.0, 1.0, 0.0)) -> 'np.ndarray'
    OpenGL/Maya-style camera-to-world matrix pointing ``eye`` at ``target``.
render(mesh: 'Optional[MeshData]' = None, uv: 'Optional[UVData]' = None, texture: 'Optional[Union[str, np.ndarray]]' = None, camera_matrix: 'Optional[np.ndarray]' = None, angle_of_view: 'float' = 35.0, resolution: 'Tuple[int, int]' = (512, 384), point_light: 'Optional[Union[dict, List[dict]]]' = None, background: 'Tuple[float, float, float, float]' = (0.0, 0.0, 0.0, 0.0), sample_method: 'Union[SampleMethod, str]' = <SampleMethod.BILINEAR: 'bilinear'>, ambient: 'float' = 0.1, wrap: 'str' = 'repeat', twosided: 'bool' = True, autofit: 'bool' = False, fit_padding: 'Optional[float]' = None, return_depth: 'bool' = False, samples_per_pixel: 'int' = 1, base_color: 'Tuple[float, float, float]' = (0.7, 0.7, 0.7), scene: "Optional[Union[List[dict], 'Scene']]" = None, default_light: 'bool' = True) -> 'Frame'
    Render a textured mesh / scene to a :class:`Frame`.
```

## `cgmath.render.scene`

`Scene`, `Object`, `Camera`, `Light`.

> Scene-graph primitives for the raytracer.

#### `Camera(TransformData)`

Maya/OpenGL-style camera: looks down its own -Z axis.

```text
Camera(name: 'str' = 'camera', angle_of_view: 'float' = 35.0, is_orthographic: 'bool' = False, ortho_height: 'Optional[float]' = None, near_plane: 'float' = 0.0, far_plane: 'float' = inf, aspect_ratio: 'Optional[float]' = None, **transform_kwargs) -> 'None'
```

Fields:

```text
angle_of_view: float = 35.0
is_orthographic: bool = False
ortho_height: Optional[float] = None
near_plane: float = 0.0
far_plane: float = inf
aspect_ratio: Optional[float] = None
```

Methods:

```text
autofit(self, points: 'np.ndarray', width: 'int', height: 'int') -> 'None'
    Override this camera's position so that ``points`` fit the frustum.
classmethod default_view(cls, points: 'np.ndarray', name: 'str' = 'camera', angle_of_view: 'float' = 35.0) -> "'Camera'"
    Build a 3/4 elevated view Camera looking at the points' bbox center.
look_at(self, target: 'Union[Tuple[float, float, float], np.ndarray]', up: 'Tuple[float, float, float]' = (0.0, 1.0, 0.0)) -> 'None'
    Aim this camera at ``target`` from its current world position.
```

#### `Light(TransformData)`

A scene light.

```text
Light(name: 'str' = 'light', kind: 'str' = 'point', color: 'Tuple[float, float, float]' = (1.0, 1.0, 1.0), intensity: 'float' = 1.0, falloff: 'bool' = True, **transform_kwargs) -> 'None'
```

Fields:

```text
kind: str = 'point'
color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
intensity: float = 1.0
falloff: bool = True
```

Properties:

```text
direction                          World-space forward direction (-Z axis) for infinite lights.
position                           World-space position derived from :attr:`world_matrix`.
```

Methods:

```text
as_dict(self) -> 'dict'
    Returns a dict matching the renderer's existing point_light shape.
```

#### `Object(TransformData)`

A renderable mesh with attached UVs, texture and shader options.

```text
Object(name: 'str' = 'object', mesh: 'Optional[MeshData]' = None, uv: 'Optional[UVData]' = None, texture: 'Optional[Union[str, np.ndarray]]' = None, skin: 'Optional[SkinDeformData]' = None, base_color: 'Tuple[float, float, float]' = (0.7, 0.7, 0.7), sample_method: 'str' = 'bilinear', wrap: 'str' = 'repeat', ambient: 'float' = 0.8, twosided: 'bool' = True, cast_shadows: 'bool' = True, resolution: 'Tuple[int, int]' = (500, 500), samples_per_pixel: 'Optional[int]' = 4, wireframe: 'bool' = False, wireframe_color: 'Tuple[int, int, int]' = (0, 0, 0), wireframe_thickness: 'int' = 1, background: 'Optional[Union[Tuple[float, ...], List[float]]]' = None, **transform_kwargs) -> 'None'
```

Properties:

```text
ambient                            
background                         Per-Object preview RGBA background in [0, 1].
base_color                         
buffer                             Convenience accessor for ``self.frame.array`` -- the pixel
cast_shadows                       
frame                              Cached most-recent render output, or ``None`` when the cache
mesh                               
resolution                         Per-Object preview render resolution ``(width, height)``.
sample_method                      
samples_per_pixel                  Per-Object preview MSAA sample count.
skin                               
texture                            
twosided                           
uv                                 
wireframe                          
wireframe_color                    
wireframe_thickness                
world_points                       Mesh points transformed to world space via :attr:`world_matrix`.
wrap                               
```

Methods:

```text
animate(self, clip, output: 'str' = 'animation.{frame:04d}.png', resolution: 'Optional[Tuple[int, int]]' = None, samples_per_pixel: 'Optional[int]' = None, fps: 'Optional[int]' = None, **animate_kwargs) -> 'Union[str, List[str]]'
    Quick standalone render of this Object animated by *clip*.
extract_texture_from_fbx(self, file_path: 'str', mesh_index: 'int' = 0, material_index: 'int' = 0) -> 'bool'
    Extract the diffuse (base color) texture from an FBX file and
extract_texture_from_glb(self, file_path: 'str', material_index: 'int' = 0) -> 'bool'
    Extract the PBR base color (diffuse / albedo) texture from a
classmethod from_dict(cls, data: 'dict') -> "'Object'"
    Reconstruct an Object from :meth:`to_dict`'s output.
get_loaded_texture(self) -> 'Optional[np.ndarray]'
    Returns the texture as a contiguous ``(H, W, 3) float32`` array,
imshow(self) -> 'None'
    Show the rendered frame in a window via OpenCV.
classmethod load_fbx(cls, file_path: 'str', index: 'int' = 0, extract_texture: 'bool' = True, load_skin: 'bool' = True, **object_kwargs) -> "'Object'"
    Load a mesh from an FBX file as an :class:`Object`.
classmethod load_glb(cls, file_path: 'str', index: 'int' = 0, scale_factor: 'float' = 100.0, extract_texture: 'bool' = True, load_skin: 'bool' = True, **object_kwargs) -> "'Object'"
    Load a mesh from a GLB file as an :class:`Object`.
classmethod load_obj(cls, file_path: 'str', index: 'int' = 0, **object_kwargs) -> "'Object'"
    Load a mesh from an OBJ file as an :class:`Object`.
merge(self) -> 'None'
    Merges overlapping points (and resulting overlapping faces),
pose(self, pose) -> "'Object'"
    Deform :attr:`mesh` for a posed skeleton, in place on this Object.
quadrangulate(self, source: 'str' = 'uv') -> 'None'
    Merges pairs of triangles into quads, in-place, on this
render(self, output: 'Optional[str]' = None, resolution: 'Optional[Tuple[int, int]]' = None, samples_per_pixel: 'Optional[int]' = None, **render_kwargs) -> "'Frame'"
    Quick standalone preview render of this Object.
restore_bind_pose(self) -> "'Object'"
    Put :attr:`mesh` back to the undeformed bind pose.
to_dict(self) -> 'dict'
    Serializable dict for :meth:`save` (npz / json / pkl).
to_image(self) -> "'Image'"
    Returns the rendered frame as a PIL Image.
triangulate(self, ngons_only=False) -> 'None'
    Triangulates polygons (quads / n-gons) into triangles, in-place,
turntable(self, output: 'str' = 'turntable.{frame:04d}.png', n_frames: 'int' = 120, resolution: 'Optional[Tuple[int, int]]' = None, samples_per_pixel: 'Optional[int]' = None, fps: 'int' = 30, **render_kwargs) -> 'Union[str, List[str]]'
    Quick standalone turntable of this Object.
```

#### `Scene(HierarchyData)`

A hierarchical container of Objects, Cameras and Lights.

```text
Scene(name: 'str' = 'scene', aspect_ratio: 'Optional[float]' = None, default_camera_name: 'Optional[str]' = None, resolution: 'Optional[Tuple[int, int]]' = (500, 500), samples_per_pixel: 'Optional[int]' = None, background: 'Optional[Union[Tuple[float, ...], List[float]]]' = None, autofit: 'Optional[bool]' = None, default_light: 'Optional[bool]' = None, return_depth: 'Optional[bool]' = None, angle_of_view: 'Optional[float]' = None, fit_padding: 'Optional[float]' = None) -> 'None'
```

Properties:

```text
background                         Optional RGBA background color in [0, 1] used by :meth:`render`.
buffer                             Convenience accessor for ``self.frame.array`` -- the pixel
cameras                            
frame                              Cached most-recent render output, or ``None`` when the cache
lights                             
objects                            
scene_name                         Returns the scene's own label (distinct from the inherited
```

Methods:

```text
animate(self, clip, output_pattern: 'str' = 'animation.{frame:04d}.png', fit: 'str' = 'auto', rotation_axis: 'str' = 'y', start_angle: 'float' = 0.0, end_angle: 'float' = 0.0, camera_name: 'Optional[str]' = None, fps: 'Optional[int]' = None, background: 'Optional[Tuple[float, float, float]]' = None, n_pose_samples: 'Optional[int]' = None, n_fit_samples: 'Optional[int]' = None, verbose: 'bool' = False, **render_kwargs) -> 'Union[str, List[str]]'
    Render a :class:`ClipData` frame by frame, deforming as it goes.
append(self, node: 'TransformData') -> 'None'
    S.append(value) -- append value to the end of the sequence
configure(self, **kwargs) -> "'Scene'"
    Bulk-set persistent render defaults; ``None`` values are ignored.
get_camera(self, name: 'Optional[str]' = None) -> 'Optional[Camera]'
    Returns the named Camera, the default Camera, or the first one.
imshow(self, resolution: 'Tuple[int, int]' = (500, 500)) -> 'None'
    Show the rendered frame in a window via OpenCV.
insert(self, index: 'int', node: 'TransformData') -> 'None'
    S.insert(index, value) -- insert value before index
classmethod load_glb(cls, file_path: 'str', name: 'Optional[str]' = None, scale_factor: 'float' = 100.0, load_skin: 'bool' = True, extract_texture: 'bool' = True) -> "'Scene'"
    Load every mesh in a GLB file into a new :class:`Scene` as
classmethod load_obj(cls, file_path: 'str', name: 'Optional[str]' = None) -> "'Scene'"
    Load every mesh in an OBJ file into a new :class:`Scene` as
render(self, **kwargs)
    Convenience wrapper around the module-level ``render(scene=self, ...)``.
to_image(self, resolution: 'Tuple[int, int]' = (500, 500)) -> "'Image'"
    Returns the rendered frame as a PIL Image.
turntable(self, output_pattern: 'str' = 'turntable.{frame:04d}.png', n_frames: 'int' = 120, rotation_axis: 'str' = 'y', start_angle: 'float' = 0.0, end_angle: 'float' = 360.0, fit: 'str' = 'auto', camera_name: 'Optional[str]' = None, fps: 'int' = 30, background: 'Optional[Tuple[float, float, float]]' = None, n_fit_samples: 'Optional[int]' = None, verbose: 'bool' = False, **render_kwargs) -> 'Union[str, List[str]]'
    Render a turntable animation: orbit a camera around the scene.
union_aabb(self, objects: 'Optional[List[Object]]' = None) -> 'Tuple[np.ndarray, np.ndarray]'
    Returns ``(min_corner, max_corner)`` of the world-space AABB
union_points(self, objects: 'Optional[List[Object]]' = None) -> 'np.ndarray'
    Returns the concatenation of all visible Objects' world-space
```

## `cgmath.render.frame`

the `Frame` result and its encoders.

> Render result wrapper.

#### `Frame`

One rendered image (plus optional depth buffer).

```text
Frame(array: 'np.ndarray', depth: 'Optional[np.ndarray]' = None, camera_matrix: 'Optional[np.ndarray]' = None, angle_of_view: 'Optional[float]' = None) -> None
```

Fields:

```text
array: np.ndarray
depth: Optional[np.ndarray] = None
camera_matrix: Optional[np.ndarray] = None
angle_of_view: Optional[float] = None
```

Properties:

```text
channels                           Number of colour channels (3 for RGB, 4 for RGBA).
dtype                              ``dtype`` of :attr:`array`.
has_depth                          Whether a depth buffer is attached.
height                             Image height in pixels.
image                              Lazily-built PIL ``Image`` view of :attr:`array`.
shape                              Shape of :attr:`array`.
width                              Image width in pixels.
```

Methods:

```text
classmethod encode(cls, frames: "Iterable['Frame.FrameLike']", output: 'str', fps: 'int' = 30, **kwargs: 'Any') -> 'str'
    Dispatch to :meth:`encode_mp4` / :meth:`encode_avi` /
classmethod encode_avi(cls, frames: "Iterable['Frame.FrameLike']", output: 'str', fps: 'int' = 30, overwrite: 'bool' = True) -> 'str'
    Encode RGBA frames to an alpha-preserving AVI via the PNG codec.
classmethod encode_gif(cls, frames: "Iterable['Frame.FrameLike']", output: 'str', fps: 'int' = 30, use_gifski: 'Optional[bool]' = None, overwrite: 'bool' = True) -> 'str'
    Encode a sequence of frames to a high-quality GIF.
classmethod encode_mp4(cls, frames: "Iterable['Frame.FrameLike']", output: 'str', fps: 'int' = 30, background: 'Tuple[float, float, float]' = (0.0, 0.0, 0.0), crf: 'int' = 18, preset: 'str' = 'slow', overwrite: 'bool' = True) -> 'str'
    Encode a sequence of frames to an H.264 mp4 (YouTube-compatible).
save(self, path: 'str', **kwargs: 'Any') -> 'None'
    Save the frame to disk via PIL.
wireframe(self, source: 'Any', color: 'Tuple[int, int, int, int]' = (0, 0, 0, 200), width: 'int' = 1, unique_edges: 'bool' = True, cull_backfaces: 'bool' = True) -> "'Frame'"
    Draw mesh edges over this frame as a wireframe overlay.
```

## `cgmath.render.camera`

camera math.

> Camera math helpers (look-at, default 3/4 view, perspective autofit,

Functions:

```text
look_at(eye: 'Tuple[float, float, float]', target: 'Tuple[float, float, float]', up: 'Tuple[float, float, float]' = (0.0, 1.0, 0.0)) -> 'np.ndarray'
    OpenGL/Maya-style camera-to-world matrix pointing ``eye`` at ``target``.
```

## `cgmath.render.raytracer`

the render call.

> Simple textured raytracer for ``MeshData`` / ``UVData`` pairs.

Functions:

```text
render(mesh: 'Optional[MeshData]' = None, uv: 'Optional[UVData]' = None, texture: 'Optional[Union[str, np.ndarray]]' = None, camera_matrix: 'Optional[np.ndarray]' = None, angle_of_view: 'float' = 35.0, resolution: 'Tuple[int, int]' = (512, 384), point_light: 'Optional[Union[dict, List[dict]]]' = None, background: 'Tuple[float, float, float, float]' = (0.0, 0.0, 0.0, 0.0), sample_method: 'Union[SampleMethod, str]' = <SampleMethod.BILINEAR: 'bilinear'>, ambient: 'float' = 0.1, wrap: 'str' = 'repeat', twosided: 'bool' = True, autofit: 'bool' = False, fit_padding: 'Optional[float]' = None, return_depth: 'bool' = False, samples_per_pixel: 'int' = 1, base_color: 'Tuple[float, float, float]' = (0.7, 0.7, 0.7), scene: "Optional[Union[List[dict], 'Scene']]" = None, default_light: 'bool' = True) -> 'Frame'
    Render a textured mesh / scene to a :class:`Frame`.
```

## `cgmath.render.texture`

texture I/O and sampling.

> Texture I/O + bilinear sampling helpers.

## `cgmath.formats.glb`

glTF / GLB reading and payload blobs (needs `pygltflib`).

#### `Animation(tuple)`

Animation(samplers, channels, duration)

```text
Animation(samplers, channels, duration)
```

#### `AnimationChannel(tuple)`

AnimationChannel(sampler, node, path)

```text
AnimationChannel(sampler, node, path)
```

#### `AnimationSampler(tuple)`

AnimationSampler(interpolation, keyframe_times, keyframe_values)

```text
AnimationSampler(interpolation, keyframe_times, keyframe_values)
```

#### `GlbData`

A class to hold the data from a gltf file

```text
GlbData(fname: str)
```

Properties:

```text
animations                         
mesh_list                          
```

Methods:

```text
clear_animations(self)
extract_payload(self, path: str, name='runtime_rig_retargeting.zip')
    extracts the zipped payload from the glb to disk
get_binary(self, name='runtime_rig_retargeting.zip')
inject_payload(self, path: str, name='runtime_rig_retargeting.zip')
    injects the zipped payload from disk into the glb
load(self, fname: str)
save(self, fname: str)
set_binary(self, data, name='runtime_rig_retargeting.zip')
```

#### `Mesh(tuple)`

Mesh(name, primitives)

```text
Mesh(name, primitives)
```

#### `Model(tuple)`

Model(name, nodes, ordered_node_indexes, meshes, animations, skins)

```text
Model(name, nodes, ordered_node_indexes, meshes, animations, skins)
```

#### `Primitive(tuple)`

Primitive(name, material, triangles, vertices, normals, uvs, joints, weights)

```text
Primitive(name, material, triangles, vertices, normals, uvs, joints, weights)
```

#### `Skin(tuple)`

Skin(joints, inverse_bind_matrices)

```text
Skin(joints, inverse_bind_matrices)
```

Functions:

```text
get_binary(gltf, name='runtime_rig_retargeting.zip')
    gets a binary blob from the gltf
get_dtype_cnt(accessor)
load_accessor_data(gltf, accessor)
load_gltf(fname)
    reads a gltf or glb, repairing null skin joints if that is what stops it
load_model(fname)
order_nodes_root_first(nodes)
    Returns the nodes sorted so that the parents come first. This helps make transforming bone chain hierarchies trivial.
repair_null_skin_joints(data)
    blanks out null entries in a glb's ``skins[].joints``, or None
set_binary(gltf, binary_data, name='runtime_rig_retargeting.zip')
    sets/adds a binary blob to the gltf
```

Constants:

```text
BYTE = 5120
FLOAT = 5126
SCALAR = 'SCALAR'
SHORT = 5122
UNSIGNED_BYTE = 5121
UNSIGNED_INT = 5125
UNSIGNED_SHORT = 5123
```

## `cgmath.formats.fbx`

FBX scenes, takes, layers, curves, export (needs the Autodesk FBX SDK).

#### `BaseData`

Base class for FBX data objects with common functionality

```text
BaseData(scene, data_object)
```

Properties:

```text
name                               Get the data object's name in Maya-friendly format
```

#### `BaseList`

Base class for list containers with custom access methods

```text
BaseList(data_objects=None)
```

Methods:

```text
append(self, data_object)
    Add a data object to the list
delete(self, name: str)
    Delete an object from the scene and remove it from this list
```

#### `CurveData(BaseData)`

Class to manipulate a specific animation curve from FBX data

```text
CurveData(scene, anim_curve, curve_name)
```

Properties:

```text
frames                             Get all keyframe times from the animation curve expressed in frames
key_count                          Get the number of keyframes in the animation curve
name                               Get the curve's identifier name in Maya-friendly format
times                              Get all keyframe times from the animation curve
values                             Get all keyframe values from the animation curve
```

Methods:

```text
clear(self)
    Remove all keyframes from the animation curve while preserving attribute connections
```

#### `CurveList(BaseList)`

Class to hold a list of CurveData objects with custom access methods

```text
CurveList(curve_data_objects=None)
```

Methods:

```text
delete(self, name: str)
    Delete a curve from the scene and remove it from this list
```

#### `FbxExporter`

Export fbx files from pipeline components

```text
FbxExporter()
```

Properties:

```text
manager                            fbx.FbxManager: the fbx manager that handles the memory
scene                              fbx.FbxScene: the fbx scene to export
```

Methods:

```text
add_skeleton(self, skeleton_component) -> None
    Add a skeleton component to the scene.
export(self, path: pathlib.Path, as_ascii=False, zero_root=False) -> None
    Export the scene to the given path
```

#### `LayerData(BaseData)`

Class to manipulate a specific animation layer in an FBX scene

```text
LayerData(scene, anim_layer)
```

Properties:

```text
curves                             Get all animation curves in this layer
```

Methods:

```text
create_curve(self, node_name, property_name)
    Create a new animation curve for an unkeyed attribute
rename(self, name: str)
    Rename this layer to a new name
```

#### `LayerList(BaseList)`

Class to hold a list of LayerData objects with custom access methods

```text
LayerList(layer_data_objects=None)
```

#### `SceneData(BaseData)`

Class to manipulate animation data in FBX files from Autodesk Maya

```text
SceneData(filename=None)
```

Properties:

```text
fps                                Get the scene's frames per second value
linear_units                       Get the scene's linear units
name                               Get the scene's name
nodes                              Get all nodes in the scene, including blendshape channels
scale_factor                       Get the scene's global scale factor
scene                              Get the FBX scene object
takes                              Get all takes (animation stacks) in the scene
```

Methods:

```text
create_take(self, name: str)
    Create a new take (animation stack) in the scene with a default "BaseLayer"
destroy(self)
    Clean up FBX manager and scene
load(self, filename)
    Load an FBX file and store its scene data
rename(self, name: str)
    Rename this scene to a new name
save(self, filename=None, embed_media=True, file_format=-1)
    Save the FBX scene to disk
```

#### `TakeData(BaseData)`

Class to manipulate a specific take (animation stack) in an FBX scene

```text
TakeData(scene, anim_stack)
```

Properties:

```text
curves                             Get all animation curves from this take
duration                           Get the duration of this take in seconds
end_frame                          Get the end time of this take expressed in frames
end_time                           Get the end time of this take
frame_count                        Get the duration of this take in number of frames
layers                             Get all animation layers from this take
start_frame                        Get the start time of this take expressed in frames
start_time                         Get the start time of this take
```

Methods:

```text
create_curve(self, node_name, property_name)
    Create a new animation curve for this take
create_layer(self, name: str)
    Create a new animation layer under this take
rename(self, name: str)
    Rename this take to a new name
```

#### `TakeList(BaseList)`

Class to hold a list of TakeData objects with custom access methods

```text
TakeList(take_data_objects=None)
```

Constants:

```text
LOGGER = <Logger cgmath.formats.fbx (WARNING)>
```

## `cgmath.formats.usd.stage`

USD stages, caches, sublayers (needs `pxr`).

> UsdStage utilities.

Functions:

```text
add_sublayers(stage: 'Usd.Stage', sublayer_paths: 'str | list[str]', start_dir: 'str | None' = None, replace: 'bool' = False) -> 'None'
    Adds sublayers to a given stage.
clear_missing_sublayers(stage: 'Usd.Stage') -> 'None'
    Clear missing sublayers in the given stage.
clear_sublayers(stage: 'Usd.Stage') -> 'None'
    Clear sublayers in the given stage.
get_sublayers(stage: 'Usd.Stage', absolute: 'bool' = False) -> 'list[str]'
    Returns a list of sublayers in a given stage.
new_stage(file_path: 'str', cache: 'Usd.StageCache | None' = None, no_cache: 'bool' = False) -> 'Usd.Stage'
    Creates a new stage at a given file path. If the file already exists, opens it
new_stage_in_memory(cache: 'Usd.StageCache | None' = None, no_cache: 'bool' = False) -> 'Usd.Stage'
    Creates a new stage in memory.
no_stage_cache()
    A context where all stages created/opened within are registered to a
open_stage(file_path: 'str', cache: 'Usd.StageCache | None' = None, no_cache: 'bool' = False) -> 'Usd.Stage'
    Opens a stage.
set_sublayers(stage: 'Usd.Stage', sublayer_paths: 'str | list[str]', start_dir: 'str | None' = None) -> 'None'
    Sets the sublayers in a given stage.
write_stage(stage: 'Usd.Stage', file_path: 'os.PathLike | str | None' = None, logger: 'logging.Logger | None' = None) -> 'None'
    Writes a stage to a file.
```

## `cgmath.formats.usd.prim`

USD prim search, duplication, composition (needs `pxr`).

> UsdPrim utilities.

Functions:

```text
add_api_schema(prim: 'Usd.Prim', schema: 'str') -> 'None'
    Adds an api schema to a given prim.
add_inherits(prim: 'Usd.Prim', inherits: 'str | Usd.Prim | list[Any]', replace: 'bool' = False) -> 'None'
    Adds inherit(s) to a given prim. Existing inherits are ignored.
add_references(prim: 'Usd.Prim', ref_paths: 'Any', start_dir: 'str | None' = None, replace: 'bool' = False) -> 'None'
    Adds one or more references to a given prim. If reference paths are
clear_inherits(prim: 'Usd.Prim') -> 'None'
    Clear inheritances in the given prim.
clear_references(prim: 'Usd.Prim') -> 'None'
    Clear references in the given prim.
copy_properties(src_prim: 'Usd.Prim', dst_prim: 'Usd.Prim', src_names: 'list[str] | dict[str:str] | None' = None, exclude_names: 'list[str] | None' = None)
    Copies properties from a source prim to a destination prim.
copy_property(src_prim: 'Usd.Prim', dst_prim: 'Usd.Prim', src_name: 'str', dst_name: 'str | None' = None)
    Copies a property from a source prim to a destination prim.
create_scopes(stage: 'Usd.Stage', prim_path: 'str', as_scope: 'bool' = True) -> 'Usd.Prim'
    Recursively creates scope prims to satisfy a given prim path.
duplicate_prim(prim: 'Usd.Prim', stage: 'Usd.Stage', path: 'str', prim_type: 'str | None' = None, extra_prims: 'Usd.Prim | list[Usd.Prim] | None' = None, exclude_attrs: 'list[str] | str | None' = None, attr_rename: 'Callable[[str], str] | None' = None) -> 'Usd.Prim'
    Duplicates a prim. This does a basic copy operation on the prim type
find_prims(stage: 'Usd.Stage', prim_type: '_PRIM_TYPES | None' = None, prim_name: 'str | None' = None, first_only: 'bool' = False, skip_children_callback: 'Callable | None' = None, match_prim_path: 'str | None' = None) -> 'list[Usd.Prim] | Usd.Prim | None'
    Returns a list of prims that meet the given criteria.
get_api_schemas(prim: 'Usd.Prim') -> 'list[str]'
    Returns a list of api schemas attached to a given prim.
get_inherits(prim: 'Usd.Prim') -> 'list[str]'
    Returns a list of inheritance paths in the given prim.
get_references(prim: 'Usd.Prim', abspath: 'bool' = True) -> 'list[str]'
    Returns a list of paths to referenced USD files in the given prim.
is_descendant(stage: 'Usd.Stage', prim: 'Union[Usd.Prim, str]', ancestor: 'Union[Usd.Prim, str]') -> 'bool'
    # (TODO) Not fully tested
is_prim_type(prim, prim_types: '_PRIM_TYPES') -> 'bool'
    Checks if a prim matches the given types.
is_valid_prim(file_path: 'os.PathLike | str', prim_path: 'os.PathLike | str | None') -> 'bool'
    Checks if a prim path exists in a given usd file.
iter_prims(obj: 'Usd.Stage | Usd.Prim', prim_type: '_PRIM_TYPES | None' = None, prim_name: 'str | None' = None, first_only: 'bool' = False, skip_children_callback: 'Callable | None' = None, match_prim_path: 'str | None' = None) -> 'Iterator[Usd.Prim]'
    Iterates over prims that meet the given criteria.
set_inherits(prim: 'Usd.Prim', inherits: 'str | list[str]') -> 'None'
    Sets the inherit(s) of a given prim. Existing inherits will be replaced.
set_references(prim: 'Usd.Prim', ref_paths: 'Any', start_dir: 'str | None' = None) -> 'None'
    Sets the references of a given prim. Existing references will be replaced.
```

Constants:

```text
LOGGER = <Logger cgmath.formats.usd.prim (WARNING)>
```

## `cgmath.rbf._kernels`

the 21 radial basis function kernels; the package init exports nothing.

> Radial Basis Function (RBF) Kernels Module

Functions:

```text
bump(X, r=1.0)
    Bump function (smooth compactly supported kernel).
cauchy(X, epsilon=1.0)
    Cauchy kernel (Rational Quadratic with alpha=1).
cubic(X, epsilon=1.0)
    Cubic kernel.
gaussian(X, epsilon=1.0)
    Gaussian (RBF/Squared Exponential) kernel.
get_kernel(name)
    Get a kernel function by name.
inverse_multiquadric(X, epsilon=1.0)
    Inverse Multiquadric kernel.
inverse_quadratic(X, epsilon=1.0)
    Inverse Quadratic kernel.
linear(X, epsilon=1.0)
    Linear kernel.
list_kernels()
    List all available kernel names.
log_kernel(X, epsilon=1.0)
    Logarithmic kernel.
matern_12(X, epsilon=1.0)
    Matern kernel with nu = 1/2 (Exponential kernel).
matern_32(X, epsilon=1.0)
    Matern kernel with nu = 3/2.
matern_52(X, epsilon=1.0)
    Matern kernel with nu = 5/2.
multiquadric(X, epsilon=1.0)
    Multiquadric kernel.
polyharmonic(X, k=2, epsilon=1.0)
    Generalized Polyharmonic Spline kernel.
quintic(X, epsilon=1.0)
    Quintic kernel.
thin_plate_spline(X, epsilon=1.0)
    Thin Plate Spline kernel.
wendland_c0(X, r=1.0)
    Wendland C0 (d <= 3) compactly supported kernel.
wendland_c2(X, r=1.0)
    Wendland C2 (d <= 3) compactly supported kernel.
wendland_c4(X, r=1.0)
    Wendland C4 (d <= 3) compactly supported kernel.
wendland_c6(X, r=1.0)
    Wendland C6 (d <= 3) compactly supported kernel.
wu_c2(X, r=1.0)
    Wu C2 compactly supported kernel.
wu_c4(X, r=1.0)
    Wu C4 compactly supported kernel.
```

Constants:

```text
KERNEL_REGISTRY = {'gaussian': <function gaussian at 0x000001C2AF668180>, '...
```

## `cgmath.constraints`

Procrustes alignment.

> Move all data types into the same namespace

Re-exported classes:

```text
ProcrustesData
```
