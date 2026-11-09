"""Filled-in examples of the render classes, for test_roundtrip.

The one builder, ``build_scene``, returns a Scene holding every fbx field
with every saved field away from its default where that is valid: a
hierarchy of groups, mesh nodes, camera nodes and light nodes, parented and
with every node channel set; a clip over it; the hero mesh with its UV set,
a skin, a textured material on some faces and a plain one on the rest, two
blend shape targets (one an in-between); a crate mesh; two cameras; two
lights; and every render setting changed.

The other texture dtypes a material keeps (uint16, float32, grey (H, W, 1)
and RGBA (H, W, 4)) are checked in test_render_roundtrip_extra.py.
"""

from __future__ import annotations

import numpy as np
from cgmath.geometry import CameraData, LightData, MaterialData, MorphData, SkinData
from cgmath.geometry.mesh import MeshData, UVData
from cgmath.hierarchy import ClipData, TransformData
from cgmath.render.scene import Scene

# fixed uuids: a node appended without one gets a random uuid4
_UUIDS = {
    name: f"00000000-0000-4000-8000-{index:012d}"
    for index, name in enumerate(
        ("set", "props", "hero", "crate", "rig", "front", "side", "key", "fill", "root", "spine", "head"), 1
    )
}

# the shapes' uuids
_SHAPES = {
    name: f"00000000-0000-4000-8000-{index:012d}"
    for index, name in enumerate(("hero_geo", "crate_geo", "atlas", "skin", "shot", "top", "sun", "lamp"), 101)
}


def cube_mesh(name: str = "hero_geo") -> MeshData:
    """a quad cube with hard-edged shading normals and an empty hole list"""
    points = np.array(
        [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        ]
    )
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    mesh = MeshData(
        indices = np.array(quads, dtype=np.int32).ravel(),
        counts  = np.full(6, 4, dtype=np.int32),
        points  = points,
        name    = name,
    )
    mesh.set_normals(hard_edge_angle=30.0)  # face-varying: normals + normal_indices
    mesh.hole_faces   = np.zeros(0, dtype=np.int32)  # no holes, said explicitly
    mesh.hole_counts  = np.zeros(0, dtype=np.int32)
    mesh.hole_indices = np.zeros(0, dtype=np.int32)
    return mesh


def cube_uv(mesh_uuid=None, name: str = "atlas") -> UVData:
    """the cube's UVs, one island per face in a 3 x 2 atlas, with its own
    bake settings"""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    tiles   = [(col, row) for row in range(2) for col in range(3)]
    points  = np.concatenate([(corners + tile) / (3.0, 2.0) for tile in tiles])
    uv = UVData(
        indices   = np.arange(24, dtype=np.int32),
        counts    = np.full(6, 4, dtype=np.int32),
        points    = points,
        name      = name,
        mesh_uuid = mesh_uuid,
    )
    uv.resolution = (256, 128)
    uv.antialias  = True
    return uv


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
    """the channels set_parent() leaves alone, set once the node sits in
    place, and hidden; joint_orient to draw_style included: TransformData
    saves them for every node type"""
    node.rotate_order             = rotate_order
    node.rotate_axis              = [0.0, 5.0, 0.0]
    node.joint_orient             = [0.0, 0.0, 15.0]
    node.shear                    = [0.125, 0.0, -0.25]
    node.visibility               = False
    node.segment_scale_compensate = True
    node.radius                   = 0.25
    node.draw_style               = 2


def build_scene(texture=None) -> Scene:
    """the stage: a set group over a props group holding the hero and crate
    nodes, a camera rig holding two camera nodes, a key light node under the
    side camera, a fill light node at the root, and the hero's joints; a two
    frame clip over them; the shapes on their nodes; every setting changed.
    ``texture`` replaces the hero's uint8 checker."""
    scene = Scene(
        "stage",
        default_camera      = "top",
        aspect_ratio        = 1.6,
        resolution          = (320, 200),
        samples_per_pixel   = 9,
        background          = (0.1, 0.2, 0.3),
        autofit             = False,
        default_light       = False,
        return_depth        = True,
        angle_of_view       = 40.0,
        fit_padding         = 1.25,
        base_color          = (0.2, 0.4, 0.6),
        sample_method       = "bezier",
        ambient             = 0.35,
        twosided            = False,
        cast_shadows        = False,
        wireframe           = True,
        wireframe_color     = (255, 128, 0),
        wireframe_thickness = 2,
    )

    group = TransformData(name="set",   uuid=_UUIDS["set"],   translate=[0.0, -1.0, 0.0])
    props = TransformData(name="props", uuid=_UUIDS["props"], rotate=[0.0, 45.0, 0.0])
    rig   = TransformData(name="rig",   uuid=_UUIDS["rig"],   scale=[2.0, 2.0, 2.0])
    hero  = TransformData(name="hero", uuid=_UUIDS["hero"], translate=[1.0, 2.0, 3.0], rotate=[10.0, 20.0, 30.0], scale=[1.5, 1.5, 1.5])
    crate = TransformData(name="crate", uuid=_UUIDS["crate"], translate=[-2.0, 0.5, 0.0])
    front = TransformData(name="front", uuid=_UUIDS["front"], translate=[0.0, 1.0, 10.0])
    side  = TransformData(name="side", uuid=_UUIDS["side"], translate=[10.0, 2.0, 0.0], rotate=[0.0, 90.0, 0.0])
    key   = TransformData(name="key",  uuid=_UUIDS["key"],  rotate=[-45.0, 30.0, 0.0])
    fill  = TransformData(name="fill", uuid=_UUIDS["fill"], translate=[-5.0, 5.0, 5.0])
    root  = TransformData(name="root", uuid=_UUIDS["root"], node_type="joint")
    spine = TransformData(name="spine", uuid=_UUIDS["spine"], node_type="joint", translate=[0.0, 0.25, 0.0])
    head  = TransformData(name="head", uuid=_UUIDS["head"], node_type="joint", translate=[0.0, 0.25, 0.0])

    for node in (group, props, hero, crate, rig, front, side, key, fill, root, spine, head):
        scene.add(node)
    props.set_parent(group)
    hero.set_parent(props)
    crate.set_parent(props)
    front.set_parent(rig)
    side.set_parent(rig)
    key.set_parent(side)
    spine.set_parent(root)
    head.set_parent(spine)

    # hero, side and key hidden; crate, front and fill keep their defaults,
    # so the left-out path loads back too
    _every_channel(hero, rotate_order=3)
    _every_channel(side, rotate_order=1)
    _every_channel(key,  rotate_order=5)
    for node in (hero, side, key):
        _user_attributes(node)

    # the takes: every node held, the spine bending over two frames
    clip                        = ClipData(scene.hierarchy, frames=2, start_frame=1001, fps=24.0, take="bend")
    clip.frames["spine"].rotate = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 30.0]])
    scene.clip                  = clip
    scene.clip.active           = "bend"

    mesh                  = cube_mesh()
    mesh.uuid             = _SHAPES["hero_geo"]
    mesh.parent_uuid      = _UUIDS["hero"]
    crate_geo             = cube_mesh("crate_geo")
    crate_geo.uuid        = _SHAPES["crate_geo"]
    crate_geo.parent_uuid = _UUIDS["crate"]
    scene.add(mesh, crate_geo)

    uv      = cube_uv(mesh.uuid)
    uv.uuid = _SHAPES["atlas"]
    scene.add(uv)

    spine_weights = np.clip(mesh.points[:, 1] + 0.5, 0.0, 1.0)
    weights       = np.stack([1.0 - spine_weights, spine_weights * 0.75, spine_weights * 0.25], axis=1)
    scene.add(SkinData(weights=weights, influences=["root", "spine", "head"], name="hero_weights", mesh_uuid=mesh.uuid))

    scene.add(
        MaterialData(
            name="skin", mesh_uuid=mesh.uuid, faces=[0, 1, 2, 3], color=[1.0, 1.0, 1.0], transparency=[0.1, 0.2, 0.3],
            ambient_color=[0.05, 0.1, 0.15], incandescence=[0.2, 0.0, 0.1], diffuse=0.6,
            texture=checker_texture() if texture is None else texture, texture_file="checker.png", uv_set="atlas",
            wrap_u="clamp", wrap_v="clamp",
        ),
        MaterialData(
            name="lid", mesh_uuid=mesh.uuid, faces=[4, 5], kind="phong", color=[0.9, 0.2, 0.1], specular_color=[0.3, 0.4, 0.5],
            cosine_power=35.0, reflectivity=0.25,
        ),
        MaterialData(name="wood", mesh_uuid=crate_geo.uuid, kind="phong", color=[0.6, 0.4, 0.2], diffuse=1.0),
    )

    scene.add(
        MorphData("puff", [[0.0, 0.0, 0.25]] * 4, [4, 5, 6, 7], mesh_uuid=mesh.uuid, weight=0.5, deformer="faceShapes"),
        MorphData("puff", [[0.0, 0.0, 0.1]] * 2, [4, 5], mesh_uuid=mesh.uuid, weight=0.5, at=0.5, deformer="faceShapes"),
    )

    shot = CameraData(
        name="shot", focal_length=50.0, horizontal_film_aperture=1.0, vertical_film_aperture=0.6, film_fit="vertical",
        film_offset=[0.1, -0.05], lens_squeeze_ratio=1.5, camera_scale=2.0, near_clip_plane=1.0, far_clip_plane=5000.0,
        uuid=_SHAPES["shot"], parent_uuid=_UUIDS["front"],
    )
    top = CameraData(
        name="top", orthographic=True, orthographic_width=12.5, uuid=_SHAPES["top"], parent_uuid=_UUIDS["side"],
    )
    scene.add(shot, top)

    scene.add(
        LightData(name="sun", kind="directional", color=[0.9, 0.8, 0.7], intensity=3.0, uuid=_SHAPES["sun"], parent_uuid=_UUIDS["key"]),
        LightData(name="lamp", kind="point", color=[0.4, 0.5, 1.0], intensity=2.5e4, decay_rate=2, uuid=_SHAPES["lamp"], parent_uuid=_UUIDS["fill"]),
    )
    return scene


EXAMPLES = {
    Scene: build_scene,
}
