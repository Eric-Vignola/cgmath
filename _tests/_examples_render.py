"""Filled-in examples of the render classes, for test_roundtrip.

Each builder returns a new object, built the way a user builds one, with
every saved field away from its default where that is valid: Object holds a
mesh, UVs, a bound skin and an array texture; Scene holds Objects, Cameras,
Lights and plain group nodes, parented, with every render setting changed.

The other texture dtypes an Object keeps (uint16, float32, a grey (H, W)
image) are checked in test_render_roundtrip_extra.py: EXAMPLES has one
builder per class.
"""

from __future__ import annotations

import numpy as np
from cgmath.geometry.deform import SkinDeformData
from cgmath.geometry.mesh import MeshData, UVData
from cgmath.geometry.skin_weights import SkinData
from cgmath.hierarchy import TransformData
from cgmath.render.scene import Camera, Light, Object, Scene

# fixed uuids: a node appended without one gets a random uuid4
_UUIDS = {
    name: f"00000000-0000-4000-8000-{index:012d}"
    for index, name in enumerate(
        ("set", "props", "hero", "crate", "rig", "front", "side", "key", "fill"), 1
    )
}


def cube_mesh(name: str = "hero_geo") -> MeshData:
    """a quad cube, offset by its matrix, with hard-edged shading normals and
    an empty hole list"""
    points = np.array(
        [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        ]
    )
    quads         = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    matrix        = np.eye(4)
    matrix[3, :3] = (0.0, 0.5, 0.0)
    mesh = MeshData(
        indices = np.array(quads, dtype=np.int32).ravel(),
        counts  = np.full(6, 4, dtype=np.int32),
        points  = points,
        name    = name,
        matrix  = matrix,
    )
    mesh.set_normals(hard_edge_angle=30.0)  # face-varying: normals + normal_indices
    mesh.hole_faces   = np.zeros(0, dtype=np.int32)  # no holes, said explicitly
    mesh.hole_counts  = np.zeros(0, dtype=np.int32)
    mesh.hole_indices = np.zeros(0, dtype=np.int32)
    return mesh


def cube_uv(name: str = "atlas") -> UVData:
    """the cube's UVs, one island per face in a 3 x 2 atlas, with its own
    bake settings"""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    tiles   = [(col, row) for row in range(2) for col in range(3)]
    points  = np.concatenate([(corners + tile) / (3.0, 2.0) for tile in tiles])
    uv = UVData(
        indices = np.arange(24, dtype=np.int32),
        counts  = np.full(6, 4, dtype=np.int32),
        points  = points,
        name    = name,
    )
    uv.resolution = (256, 128)
    uv.antialias  = True
    return uv


def bound_skin(mesh: MeshData) -> SkinDeformData:
    """a dual-quaternion skin on ``mesh``'s points, three joints stacked in y,
    bound so its compact weights are filled in"""
    joints  = ["root", "spine", "head"]
    spine   = np.clip(mesh.points[:, 1] + 0.5, 0.0, 1.0)
    weights = np.stack([1.0 - spine, spine * 0.75, spine * 0.25], axis=1)
    skin    = SkinData(weights=weights, influences=joints, name="hero_weights")

    binds          = np.repeat(np.eye(4)[None], 3, axis=0)
    binds[:, 3, 1] = (0.0, -0.25, -0.5)  # inverse of joints 0.25 apart in y
    deformer = SkinDeformData(
        mesh, skin, inverse_bind_matrices=binds, method="dqs", name="hero_skin"
    )
    deformer.bind()
    return deformer


def checker_texture() -> np.ndarray:
    """an (8, 12, 3) uint8 checker with a ramp in red"""
    texture             = np.full((8, 12, 3), 40, dtype=np.uint8)
    texture[::2, ::2]   = 220
    texture[1::2, 1::2] = 220
    texture[..., 0]     = np.linspace(0, 255, 12).astype(np.uint8)
    return texture


def _user_attributes(node: TransformData) -> None:
    """a spread of user attribute types, NaN and inf among the doubles and
    an empty array among the lists"""
    node.add_user_attribute("lod", 2)
    node.add_user_attribute("exposure", float("nan"))
    node.add_user_attribute("clip_far", float("inf"), attribute_type="doubleLinear")
    node.add_user_attribute("pass_name", "beauty")
    node.add_user_attribute("tags",      ["hero", "prop"])
    node.add_user_attribute("ramp",      [0.0, 0.25, 1.0])
    node.add_user_attribute("unused", [], attribute_type="doubleArray")
    node.add_user_attribute(
        "pivots", [[0.0, 1.0, 0.0], [0.5, 0.5, 0.5]], attribute_type="vectorArray"
    )
    node.add_user_attribute(
        "shading", "matte", attribute_type="enum", enum_names="glossy:matte:flat"
    )
    node.add_user_attribute("holdout", True, keyable=False)


def _every_channel(node: TransformData, rotate_order: int) -> None:
    """the channels a set_parent() or look_at() rewrites (it re-solves the
    local matrix, rotate_axis back to zero), set once the node sits in place,
    and hidden; joint_orient to draw_style included: TransformData saves them
    for every node type"""
    node.rotate_order             = rotate_order
    node.rotate_axis              = [0.0, 5.0, 0.0]
    node.joint_orient             = [0.0, 0.0, 15.0]
    node.visibility               = False
    node.segment_scale_compensate = True
    node.radius                   = 0.25
    node.draw_style               = 2


def _hero(texture=None) -> Object:
    """the hero Object, every setting away from its default"""
    mesh = cube_mesh()
    return Object(
        name="hero",
        mesh=mesh,
        uv=cube_uv(),
        texture=checker_texture() if texture is None else texture,
        skin=bound_skin(mesh),
        base_color=(0.2, 0.4, 0.6),
        sample_method="bezier",
        wrap="clamp",
        ambient=0.35,
        twosided=False,
        cast_shadows=False,
        resolution=(64, 32),
        samples_per_pixel=16,
        wireframe=True,
        wireframe_color=(255, 128, 0),
        wireframe_thickness=2,
        background=(0.5, 0.5, 0.5, 0.25),
        uuid=_UUIDS["hero"],
        translate=[1.0, 2.0, 3.0],
        rotate=[10.0, 20.0, 30.0],
        scale=[1.5, 1.5, 0.5],
    )


def build_scene(texture=None) -> Scene:
    """a set group holding a props group, the hero Object and a crate under
    the props, a camera rig with two Cameras, a key Light under the side
    Camera and a fill Light at the root; every Scene setting changed.
    ``texture`` replaces the hero's uint8 checker."""
    scene = Scene(
        "stage",
        aspect_ratio        = 1.6,
        default_camera_name = "side",
        resolution          = (320, 200),
        samples_per_pixel   = 9,
        background          = (0.1, 0.2, 0.3),
        autofit             = False,
        default_light       = False,
        return_depth        = True,
        angle_of_view       = 40.0,
        fit_padding         = 1.25,
    )

    group = TransformData(name="set",   uuid=_UUIDS["set"],   translate=[0.0, -1.0, 0.0])
    props = TransformData(name="props", uuid=_UUIDS["props"], rotate=[0.0, 45.0, 0.0])
    rig   = TransformData(name="rig",   uuid=_UUIDS["rig"],   scale=[2.0, 2.0, 2.0])
    hero  = _hero(texture)

    # a second, plain Object: no uv, skin or texture, most settings default
    crate = Object(
        name      = "crate",
        mesh      = cube_mesh("crate_geo"),
        uuid      = _UUIDS["crate"],
        ambient   = 0.6,
        wrap      = "clamp",
        translate = [-2.0, 0.5, 0.0],
    )
    crate.samples_per_pixel = None  # defer to the renderer

    front = Camera(name="front", uuid=_UUIDS["front"], translate=[0.0, 1.0, 10.0])
    side = Camera(
        name            = "side",
        uuid            = _UUIDS["side"],
        angle_of_view   = 50.0,
        is_orthographic = True,
        ortho_height    = 12.5,
        near_plane      = 0.1,
        far_plane       = 500.0,
        aspect_ratio    = 2.0,
        translate       = [10.0, 2.0, 0.0],
    )
    side.look_at((0.0, 0.0, 0.0))

    key = Light(
        name      = "key",
        uuid      = _UUIDS["key"],
        kind      = "infinite",
        color     = (0.9, 0.8, 0.7),
        intensity = 3.0,
        falloff   = False,
        rotate    = [-45.0, 30.0, 0.0],
    )
    fill = Light(
        name      = "fill",
        uuid      = _UUIDS["fill"],
        color     = (0.4, 0.5, 1.0),
        intensity = 2.5e4,
        translate = [-5.0, 5.0, 5.0],
    )

    for node in (group, props, hero, crate, rig, front, side, key, fill):
        scene.append(node)
    props.set_parent(group)
    hero.set_parent(props)
    crate.set_parent(props)
    front.set_parent(rig)
    side.set_parent(rig)
    key.set_parent(side)

    # hero, side and key hidden; crate, front and fill keep their defaults,
    # so the left-out path loads back too
    _every_channel(hero, rotate_order=3)
    _every_channel(side, rotate_order=1)
    _every_channel(key,  rotate_order=5)
    for node in (hero, side, key):
        _user_attributes(node)
    return scene


def build_object() -> Object:
    """the hero, saved on its own: its parent is a uuid it keeps"""
    return build_scene()["hero"]


def build_camera() -> Camera:
    """the orthographic side Camera, aimed, under the camera rig"""
    return build_scene()["side"]


def build_light() -> Light:
    """the infinite key Light, under the side Camera"""
    return build_scene()["key"]


EXAMPLES = {
    Object: build_object,
    Camera: build_camera,
    Light:  build_light,
    Scene:  build_scene,
}
