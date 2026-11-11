"""Records Maya 2025's side of cgmath's fbx tests into _tests/test_assets/maya2025.

Run under mayapy, by hand, when a Maya convention is measured again or a
test's data changes; nothing in the test suite needs Maya:

    mayapy tools/make_maya_fbx_fixtures.py [case ...]

Each case folder holds the files Maya exported (``maya*.fbx``), the files our
writer produced at recording time and Maya imported (``ours*.fbx``), and
``readings.json`` + ``readings.npz``: everything the tests used to ask Maya
through ``cmds``, as the tests in ``cgmath/_tests`` read it back through
``maya_fixtures.py``. ``manifest.json`` names the Maya that recorded them.
The Python side of each case (the data saved, the sdk files written) comes
from the test modules themselves, so a change there is recorded here.
"""
import json
import os
import shutil
import sys
import time

import maya.standalone

maya.standalone.initialize(name="python")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import maya.api.OpenMaya as om  # noqa: E402
import maya.api.OpenMayaAnim as oma  # noqa: E402
import numpy as np  # noqa: E402
from maya import cmds, mel  # noqa: E402
from PIL import Image  # noqa: E402

cmds.loadPlugin("fbxmaya", quiet=True)

from cgmath._tests import test_clip_list as clip_list  # noqa: E402
from cgmath._tests import test_fbx_axes as axes  # noqa: E402
from cgmath._tests import test_fbx_cameras_lights as views  # noqa: E402
from cgmath._tests import test_fbx_colors as colors  # noqa: E402
from cgmath._tests import test_fbx_materials as mats  # noqa: E402
from cgmath._tests import test_fbx_morphs as morphs  # noqa: E402
from cgmath._tests import test_fbx_nurbs as nurbs  # noqa: E402
from cgmath._tests import test_short_names as short_names  # noqa: E402
from cgmath._tests import test_skeleton_fbx as skel  # noqa: E402
from cgmath.formats import load_fbx, save_fbx  # noqa: E402
from cgmath.geometry import CameraList, MeshList, UVList  # noqa: E402
from cgmath.hierarchy import ClipList, HierarchyData, TransformData  # noqa: E402
from cgmath.hierarchy.hierarchy import _user_channels  # noqa: E402

FBX   = skel.FBX
OUT   = os.path.join(ROOT, "cgmath", "_tests", "test_assets", "maya2025")
START = skel.START
TAKE  = skel.TAKE


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is no json value")


class Recorder:
    """one case's folder: the files it keeps, the arrays and values it records"""

    def __init__(self, name: str):
        self.name   = name
        self.folder = os.path.join(OUT, name)
        shutil.rmtree(self.folder, ignore_errors=True)
        os.makedirs(self.folder)
        self.arrays = {}
        self.values = {}

    def path(self, filename: str) -> str:
        return os.path.join(self.folder, filename)

    def array(self, key: str, value) -> None:
        self.arrays[key] = np.asarray(value)

    def value(self, key: str, value) -> None:
        self.values[key] = value

    def save(self) -> None:
        np.savez_compressed(self.path("readings.npz"), **self.arrays)
        with open(self.path("readings.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.values, f, indent=1, default=_json, sort_keys=True)
            f.write("\n")
        print(f"{self.name}: {len(self.arrays)} arrays, {len(self.values)} values, {sorted(os.listdir(self.folder))}")


# --------------------------------------------------------------- maya side --- #
def new_scene(units: bool = True) -> None:
    cmds.file(new=True, force=True)
    if units:
        cmds.currentUnit(time="ntsc", linear="cm", angle="deg")
        cmds.upAxis(axis="y")


def imported(path: str, take=None, units: bool = True) -> None:
    """a fresh ntsc Y-up scene with the file imported; ``take`` picks one of
    the file's takes by its 1-based position (FBXImport -t)"""
    new_scene(units)
    mel.eval("FBXResetImport")
    flag = "" if take is None else f" -t {int(take)}"
    mel.eval(f'FBXImport -f "{path.replace(os.sep, "/")}"{flag}')


def exported(path: str, at=None, cameras=False, lights=False, shapes=False, embed=None) -> str:
    """Maya's FBXExport of the scene, at frame ``at`` when given"""
    if at is not None:
        cmds.currentTime(at)
    mel.eval("FBXResetExport")
    # what FBXResetExport leaves as an earlier export set it
    mel.eval("FBXExportConvertUnitString cm")
    mel.eval("FBXExportUpAxis y")
    if cameras:
        mel.eval("FBXExportCameras -v true")
    if lights:
        mel.eval("FBXExportLights -v true")
    if shapes:
        mel.eval("FBXExportShapes -v true")
    if embed is not None:
        mel.eval(f"FBXExportEmbeddedTextures -v {'true' if embed else 'false'}")
    mel.eval(f'FBXExport -f "{path.replace(os.sep, "/")}"')
    return path


def world(node: str, time=None) -> np.ndarray:
    kwargs = {} if time is None else {"time": time}
    return np.array(cmds.getAttr(f"{node}.worldMatrix", **kwargs)).reshape(4, 4)


def xform_world(node: str) -> np.ndarray:
    return np.array(cmds.xform(node, q=True, ws=True, m=True)).reshape(4, 4)


def triple(node: str, attr: str, time=None) -> list:
    kwargs = {} if time is None else {"time": time}
    return list(cmds.getAttr(f"{node}.{attr}", **kwargs)[0])


def node_meta(name: str) -> dict:
    """the node's name count, type, shape types, parent and static attributes"""
    shapes = cmds.listRelatives(name, shapes=True) or []
    parent = cmds.listRelatives(name, parent=True)
    meta = {
        "count":        len(cmds.ls(name, long=True)),
        "type":         cmds.nodeType(name),
        "shapes":       [cmds.nodeType(shape) for shape in shapes],
        "parent":       parent[0] if parent else None,
        "rotate_order": cmds.getAttr(f"{name}.rotateOrder"),
        "rotate_axis":  triple(name, "rotateAxis"),
    }
    if meta["type"] == "joint":
        meta["joint_orient"] = triple(name, "jointOrient")
        meta["ssc"]          = bool(cmds.getAttr(f"{name}.segmentScaleCompensate"))
    return meta


def user_value(node: str, name: str, time=None) -> np.ndarray:
    kwargs = {} if time is None else {"time": time}
    value  = cmds.getAttr(f"{node}.{name}", **kwargs)
    return np.ravel(value[0] if isinstance(value, list) else value).astype(np.float64)


def dag(name: str):
    shapes = cmds.listRelatives(name, shapes=True, noIntermediate=True, fullPath=True)
    chosen = om.MSelectionList()
    chosen.add(shapes[0])
    return chosen.getDagPath(0)


def mesh_points(path, space, time=None) -> np.ndarray:
    if time is not None:
        cmds.currentTime(time)
    return np.array([tuple(p)[:3] for p in om.MFnMesh(path).getPoints(space)])


def maya_uvs(name: str) -> dict:
    """{set: (face counts, (corners, 2) UVs)} of the mesh, in Maya's set order"""
    shape = om.MFnMesh(dag(name))
    found = {}
    for uv_set in shape.getUVSetNames():
        us, vs = shape.getUVs(uv_set)
        sizes, ids = shape.getAssignedUVs(uv_set)
        held          = np.stack([np.array(us), np.array(vs)], axis=1) if len(us) else np.zeros((0, 2))
        found[uv_set] = (np.array(sizes), held[np.array(ids, dtype=np.intp)] if len(ids) else np.zeros((0, 2)))
    return found


def maya_colors(shape: str) -> tuple:
    """(the mesh's colour sets, its colour a face corner in face order, (C, 4), from the current set)"""
    sets = cmds.polyColorSet(shape, query=True, allColorSets=True) or []
    held = []
    for face in range(cmds.polyEvaluate(shape, face=True)):
        for vf in cmds.ls(cmds.polyListComponentConversion(f"{shape}.f[{face}]", toVertexFace=True), flatten=True):
            held.append(cmds.polyColorPerVertex(vf, query=True, rgb=True) + cmds.polyColorPerVertex(vf, query=True, alpha=True))
    return sets, np.array(held, dtype=np.float64)


def record_nodes(rec: Recorder, nodes, frames=None, prefix: str = "") -> None:
    """every node's meta, and its channels, world and user channels static or on ``frames``"""
    rec.value(prefix + "nodes", {node.name: node_meta(node.name) for node in nodes})
    for node in nodes:
        name = node.name
        if frames is None:
            for attr in ("translate", "rotate", "scale"):
                rec.array(f"{prefix}{attr}/{name}", triple(name, attr))
            rec.array(f"{prefix}world/{name}", world(name))
            for channel in _user_channels(node.user_defined_attributes):
                rec.array(f"{prefix}user/{name}/{channel}", user_value(name, channel))
            continue
        for attr in ("translate", "rotate", "scale"):
            rec.array(f"{prefix}{attr}/{name}", [triple(name, attr, f) for f in frames])
        rec.array(f"{prefix}world/{name}", [world(name, f) for f in frames])
        for channel in _user_channels(node.user_defined_attributes):
            rec.array(f"{prefix}user/{name}/{channel}", [user_value(name, channel, f) for f in frames])


def counts() -> dict:
    return {kind: len(cmds.ls(type=kind)) for kind in ("dagPose", "animCurve", "skinCluster")}


# ------------------------------------------------------------------- cases --- #
def clip_import(rec: Recorder) -> None:
    """our clip imported: every node, every frame"""
    ours = rec.path("ours.fbx")
    save_fbx(ours, clip=skel.take(skel.skeleton()))
    back = load_fbx(ours).clip.active
    imported(ours)
    rec.value("counts", counts())
    record_nodes(rec, back, [START + f for f in range(back.frame_count)])


def visibility_locator(rec: Recorder) -> None:
    rig = skel.skeleton()
    rig["prop"].node_type = "locator"
    clip = skel.hidden(skel.take(rig))
    ours = rec.path("ours.fbx")
    save_fbx(ours, clip=clip)
    imported(ours)
    rec.value("nodes", {node.name: node_meta(node.name) for node in clip})
    for node in clip:
        rec.array(f"visibility/{node.name}", [bool(cmds.getAttr(f"{node.name}.visibility", time=START + f)) for f in range(clip.frame_count)])
    rec.value("prop_visibility_curves", len(cmds.listConnections("prop.visibility", type="animCurve") or []))
    exported(rec.path("maya.fbx"), at=START)


def rrss(rec: Recorder) -> None:
    path = skel._written(rec.path("source.fbx"), parent_scale=(2.0, 2.0, 2.0), inherit=FBX.FbxTransform.EInheritType.eInheritRrSs)
    rig  = load_fbx(path).hierarchy
    imported(path)
    for node in rig:
        rec.array(f"world/{node.name}", world(node.name))


def takes(rec: Recorder) -> None:
    ours = rec.path("ours.fbx")
    save_fbx(ours, clip=skel.takes())
    back = load_fbx(ours).clip
    for index, clip in enumerate(back):
        imported(ours, take=index + 1)
        rec.value(f"take{index}/animCurve", len(cmds.ls(type="animCurve")))
        frames = [clip.start_frame + f for f in range(clip.frame_count)]
        for node in clip:
            rec.array(f"take{index}/world/{node.name}", [world(node.name, f) for f in frames])


def skeleton_import(rec: Recorder) -> None:
    rig  = skel.skeleton()
    ours = rec.path("ours.fbx")
    save_fbx(ours, rig)
    imported(ours)
    rec.value("counts", counts())
    record_nodes(rec, rig)


def export_reads_back(rec: Recorder) -> None:
    ours = rec.path("ours.fbx")
    save_fbx(ours, clip=skel.take(skel.skeleton()))
    imported(ours)
    exported(rec.path("maya.fbx"), at=START)


def _skinned(on: str) -> dict:
    rig  = skel.placed()
    mesh = skel.body(rig[on].uuid)
    return {"hierarchy": rig, "clip": skel.motion(rig), "meshes": mesh, "skins": skel.weighed(mesh)}


def _skin_case(rec: Recorder, on: str) -> None:
    source = _skinned(on)
    ours   = rec.path("ours.fbx")
    save_fbx(ours, **source)
    imported(ours)
    path     = dag(on)
    clusters = cmds.ls(type="skinCluster")
    poses    = cmds.ls(type="dagPose")
    rec.value("skinClusters",        len(clusters))
    rec.value("history_has_cluster", cmds.ls(cmds.listHistory(path.fullPathName()), type="skinCluster") == clusters)
    rec.value("dagPoses",            len(poses))
    rec.value("bindPose",            bool(cmds.getAttr(f"{poses[0]}.bindPose")) if poses else None)
    chosen = om.MSelectionList()
    chosen.add(clusters[0])
    skin       = oma.MFnSkinCluster(chosen.getDependNode(0))
    influences = [x.partialPathName() for x in skin.influenceObjects()]
    rec.value("influences", influences)
    every = om.MFnSingleIndexedComponent()
    part  = every.create(om.MFn.kMeshVertComponent)
    every.setCompleteData(source["meshes"].point_count)
    weights, _ = skin.getWeights(path, part)
    rec.array("weights", np.array(weights).reshape(source["meshes"].point_count, len(influences)))
    rec.array("bindPreMatrix", [
        np.array(cmds.getAttr(f"{clusters[0]}.bindPreMatrix[{skin.indexForInfluenceObject(joint)}]")).reshape(4, 4)
        for joint in skin.influenceObjects()
    ])
    rec.array("geomMatrix", np.array(cmds.getAttr(f"{clusters[0]}.geomMatrix")).reshape(4, 4))
    rec.array("points", [mesh_points(path, om.MSpace.kWorld, START + f) for f in range(TAKE)])
    exported(rec.path("maya.fbx"), at=START)


def skin_geo(rec: Recorder) -> None:
    _skin_case(rec, "geo")


def skin_placed(rec: Recorder) -> None:
    _skin_case(rec, "placed")


def two_meshes(rec: Recorder) -> None:
    clip   = skel.motion(skel.bound())
    meshes = MeshList([skel.body(clip["geo"].uuid), skel.tetra(clip["prop"].uuid)])
    ours   = rec.path("ours.fbx")
    save_fbx(ours, clip=clip, meshes=meshes)
    imported(ours)
    rec.value("counts", counts())
    for name in ("geo", "prop"):
        path  = dag(name)
        shape = om.MFnMesh(path)
        rec.value(f"{name}/curves", cmds.listConnections(path.fullPathName(), type="animCurve"))
        rec.array(f"{name}/points", mesh_points(path, om.MSpace.kObject))
        normal_counts, ids = shape.getNormalIds()
        rec.array(f"{name}/normal_counts", np.array(normal_counts))
        rec.array(f"{name}/normal_ids",    np.array(ids))
        rec.array(f"{name}/normals",       np.array([tuple(n) for n in shape.getNormals()]))
        for f in (0, TAKE - 1):
            rec.array(f"{name}/world_points/{f}", mesh_points(path, om.MSpace.kWorld, START + f))


def uvs_import(rec: Recorder) -> None:
    mesh = skel.body()
    ours = rec.path("ours.fbx")
    save_fbx(ours, meshes=mesh, uvs=skel.maps(mesh))
    imported(ours)
    found = maya_uvs("body")
    rec.value("uv_sets", list(found))
    for name, (sizes, corners) in found.items():
        rec.array(f"uv/{name}/sizes", sizes)
        rec.array(f"uv/{name}/corners", corners)


def uvs_export(rec: Recorder) -> None:
    source = skel.bundle()
    ours   = rec.path("ours.fbx")
    save_fbx(ours, **source, uvs=skel.maps(source["meshes"]))
    imported(ours)
    exported(rec.path("maya.fbx"), at=START)


def uvs_none(rec: Recorder) -> None:
    ours = rec.path("ours.fbx")
    save_fbx(ours, meshes=skel.body())
    imported(ours)
    rec.value("body_sets", {name: len(corners) for name, (_, corners) in maya_uvs("body").items()})
    exported(rec.path("maya.fbx"), at=START)
    new_scene()
    cmds.polyCube(name="box", createUVs=0)
    rec.value("box_sets", {name: len(corners) for name, (_, corners) in maya_uvs("box").items()})
    exported(rec.path("box.fbx"), at=START)


def zup(rec: Recorder) -> None:
    y_cm = rec.path("y_cm.fbx")
    save_fbx(y_cm, clip=axes.turned_clip(axes.turned_rig()))
    for unit in ("cm", "m"):
        path = axes.relabelled(y_cm, rec.path(f"z_{unit}.fbx"), FBX.FbxAxisSystem.MayaZUp, unit)
        clip = load_fbx(path).clip.active
        imported(path)
        rec.value(f"{unit}/dagPoses", len(cmds.ls(type="dagPose")))
        rec.value(f"{unit}/hips_type", cmds.nodeType("hips"))
        for node in clip:
            rec.array(f"{unit}/world/{node.name}", [world(node.name, f) for f in range(axes.FRAMES)])
        rec.array(f"{unit}/hips_joint_orient", triple("hips", "jointOrient"))
        rec.array(f"{unit}/knee_joint_orient", triple("knee", "jointOrient"))
        rec.array(f"{unit}/group_rotate",      triple("group", "rotate", 0))
        rec.array(f"{unit}/loc_rotate",        triple("loc", "rotate", 0))
        rec.array(f"{unit}/hips_scale",        triple("hips", "scale"))
        rec.array(f"{unit}/knee_translate",    triple("knee", "translate"))


def zup_skinned(rec: Recorder) -> None:
    rig    = skel.bound()
    mesh   = skel.body(rig["geo"].uuid)
    source = {"hierarchy": rig, "clip": skel.motion(rig), "meshes": mesh, "skins": skel.weighed(mesh)}
    y_cm   = rec.path("y_cm.fbx")
    save_fbx(y_cm, **source)
    path = axes.relabelled(y_cm, rec.path("z_cm.fbx"), FBX.FbxAxisSystem.MayaZUp, "cm")
    imported(path)
    clusters = cmds.ls(type="skinCluster")
    rec.value("skinClusters", len(clusters))
    rec.array("geomMatrix", np.array(cmds.getAttr(f"{clusters[0]}.geomMatrix")).reshape(4, 4))
    held = dag("geo")
    for f in (0, TAKE // 2, TAKE - 1):
        rec.array(f"world_points/{f}", mesh_points(held, om.MSpace.kWorld, START + f))


CAMERA_ATTRS = ("focalLength", "horizontalFilmAperture", "verticalFilmAperture", "horizontalFilmOffset", "verticalFilmOffset",
                "lensSqueezeRatio", "nearClipPlane", "farClipPlane", "orthographicWidth", "filmFit", "orthographic", "cameraScale")
LIGHT_ATTRS = ("intensity", "decayRate", "coneAngle", "penumbraAngle", "dropoff")


def camera_reading(shape: str) -> dict:
    return {attr: cmds.getAttr(f"{shape}.{attr}") for attr in CAMERA_ATTRS}


def light_reading(shape: str) -> dict:
    held = {"type": cmds.nodeType(shape), "color": list(cmds.getAttr(f"{shape}.color")[0])}
    for attr in LIGHT_ATTRS:
        if cmds.attributeQuery(attr, node=shape, exists=True):
            held[attr] = cmds.getAttr(f"{shape}.{attr}")
    return held


def views_scene() -> dict:
    """the probe's scene: a tuned camera, an orthographic one, a fresh one
    under a turned group, every light kind turned and moved; {transform: shape}"""
    cmds.file(new=True, force=True)
    cmds.currentUnit(time="film", linear="cm", angle="deg")
    made = {}
    tuned, shape = cmds.camera(name="tunedCam")
    for attr, value in (("focalLength", 50.0), ("horizontalFilmAperture", 1.0), ("verticalFilmAperture", 0.6),
                        ("filmFit", 2), ("horizontalFilmOffset", 0.1), ("verticalFilmOffset", -0.05),
                        ("lensSqueezeRatio", 1.5), ("cameraScale", 2.0), ("nearClipPlane", 1.0), ("farClipPlane", 5000.0)):
        cmds.setAttr(f"{shape}.{attr}", value)
    cmds.setAttr(f"{tuned}.translate", 10.0, 20.0, 30.0)
    cmds.setAttr(f"{tuned}.rotate", 10.0, 20.0, 30.0)
    made[tuned] = shape
    ortho, shape = cmds.camera(name="orthoCam")
    cmds.setAttr(f"{shape}.orthographic", True)
    cmds.setAttr(f"{shape}.orthographicWidth", 50.0)
    cmds.setAttr(f"{ortho}.rotate", -90.0, 0.0, 0.0)
    made[ortho] = shape
    plain, shape = cmds.camera(name="plainCam")
    group = cmds.group(empty=True, name="camGroup")
    cmds.setAttr(f"{group}.rotate", 0.0, 45.0, 0.0)
    cmds.parent(plain, group)
    made[plain] = shape
    for kind, make in (("point", cmds.pointLight), ("directional", cmds.directionalLight), ("spot", cmds.spotLight),
                       ("ambient", cmds.ambientLight)):
        node = make(name=f"{kind}LightShape")
        if cmds.nodeType(node) != "transform":
            node = cmds.listRelatives(node, parent=True)[0]
        node  = cmds.rename(node, f"{kind}Light")
        shape = cmds.listRelatives(node, shapes=True)[0]
        cmds.setAttr(f"{shape}.color", 0.25, 0.5, 1.0)
        cmds.setAttr(f"{shape}.intensity", 2.5)
        if kind in ("point", "spot"):
            cmds.setAttr(f"{shape}.decayRate", 2)
        if kind == "spot":
            cmds.setAttr(f"{shape}.coneAngle",     50.0)
            cmds.setAttr(f"{shape}.penumbraAngle", 5.0)
            cmds.setAttr(f"{shape}.dropoff",       3.0)
        cmds.setAttr(f"{node}.translate", 1.0, 2.0, 3.0)
        cmds.setAttr(f"{node}.rotate", 10.0, 20.0, 30.0)
        made[node] = shape
    for kind in ("areaLight", "volumeLight"):
        node = cmds.shadingNode(kind, asLight=True, name=f"{kind}Shape")
        if cmds.nodeType(node) != "transform":
            node = cmds.listRelatives(node, parent=True)[0]
        node = cmds.rename(node, kind)
        cmds.setAttr(f"{node}.rotate", 10.0, 20.0, 30.0)
        made[node] = cmds.listRelatives(node, shapes=True)[0]
    return made


def views_export(rec: Recorder) -> None:
    made = views_scene()
    exported(rec.path("maya.fbx"), cameras=True, lights=True)
    rec.value("made", made)
    for node, shape in made.items():
        rec.array(f"world/{node}", xform_world(node))
    rec.value("cameras", {node: camera_reading(shape) for node, shape in made.items() if cmds.nodeType(shape) == "camera"})
    rec.value("lights", {node: light_reading(shape) for node, shape in made.items() if cmds.nodeType(shape) != "camera"})
    rec.value("fov", {
        node: {"h": cmds.camera(shape, q=True, horizontalFieldOfView=True), "v": cmds.camera(shape, q=True, verticalFieldOfView=True)}
        for node, shape in made.items() if cmds.nodeType(shape) == "camera"
    })


def views_import(rec: Recorder) -> None:
    rig     = views.views_rig()
    cameras = CameraList([views.shot(rig), views.ortho(rig)])
    lights  = views.views_lights(rig)
    ours    = rec.path("ours.fbx")
    save_fbx(ours, rig, cameras=cameras, lights=lights)
    imported(ours)
    for node in rig:
        rec.array(f"world/{node.name}", xform_world(node.name))
    held = {}
    for camera in cameras:
        node       = rig[camera.parent_uuid].name
        shapes     = cmds.listRelatives(node, shapes=True, type="camera") or []
        held[node] = {"shape_count": len(shapes)}
        if shapes:
            held[node].update(camera_reading(shapes[0]))
    rec.value("cameras", held)
    held = {}
    for light in lights:
        node       = rig[light.parent_uuid].name
        shapes     = cmds.listRelatives(node, shapes=True) or []
        held[node] = {"shape_count": len(shapes)}
        if shapes:
            held[node].update(light_reading(shapes[0]))
    rec.value("lights", held)


def other_writer(rec: Recorder) -> None:
    path = views.other_writer(rec.path("other.fbx"))
    imported(path)
    cmds.currentUnit(time="film")
    for name in ("cam", "sun"):
        rec.array(f"world/{name}", [world(name, f) for f in range(views.FRAMES)])


def materials_scene(grad: str) -> dict:
    """the probe's scene: a 2 by 2 plane, faces 0 and 1 under a lambert whose
    colour a file texture feeds (wrap u on, v off), faces 2 and 3 under a
    phong; a cube under a blinn; a sphere under a standardSurface;
    {shader: (mesh, faces or None)}"""
    cmds.file(new=True, force=True)
    cmds.currentUnit(time="ntsc", linear="cm", angle="deg")

    def shader(kind, name, **values):
        made  = cmds.shadingNode(kind, asShader=True, name=name)
        group = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=f"{name}SG")
        cmds.connectAttr(f"{made}.outColor", f"{group}.surfaceShader", force=True)
        for attr, value in values.items():
            if isinstance(value, (tuple, list)):
                cmds.setAttr(f"{made}.{attr}", *value, type="double3")
            else:
                cmds.setAttr(f"{made}.{attr}", value)
        return made, group

    plane_node, _ = cmds.polyPlane(name="plane", width=2.0, height=2.0, subdivisionsX=2, subdivisionsY=2)
    lambert, lambert_sg = shader("lambert", "skinLambert", transparency=(0.1, 0.2, 0.3), ambientColor=(0.05, 0.1, 0.15),
                                 incandescence=(0.2, 0.0, 0.1), diffuse=0.6)
    file_node = cmds.shadingNode("file", asTexture=True, name="gradFile")
    place     = cmds.shadingNode("place2dTexture", asUtility=True, name="gradPlace")
    cmds.connectAttr(f"{place}.outUV", f"{file_node}.uvCoord")
    cmds.connectAttr(f"{place}.outUvFilterSize", f"{file_node}.uvFilterSize")
    cmds.setAttr(f"{file_node}.fileTextureName", grad.replace(os.sep, "/"), type="string")
    cmds.setAttr(f"{place}.wrapU", 1)
    cmds.setAttr(f"{place}.wrapV", 0)
    cmds.connectAttr(f"{file_node}.outColor", f"{lambert}.color", force=True)
    phong, phong_sg = shader("phong", "shinyPhong", color=(0.9, 0.2, 0.1), specularColor=(0.3, 0.4, 0.5), cosinePower=35.0, reflectivity=0.25)
    cmds.sets(f"{plane_node}.f[0:1]", edit=True, forceElement=lambert_sg)
    cmds.sets(f"{plane_node}.f[2:3]", edit=True, forceElement=phong_sg)
    cube, _ = cmds.polyCube(name="cube")
    blinn, blinn_sg = shader("blinn", "dullBlinn", color=(0.1, 0.8, 0.2), eccentricity=0.45, specularRollOff=0.6, specularColor=(0.7, 0.7, 0.2), reflectivity=0.1)
    cmds.sets(cube, edit=True, forceElement=blinn_sg)
    ball, _ = cmds.polySphere(name="ball", subdivisionsX=8, subdivisionsY=6)
    standard, standard_sg = shader("standardSurface", "pbr", baseColor=(0.2, 0.3, 0.9), base=0.7)
    cmds.sets(ball, edit=True, forceElement=standard_sg)
    cmds.setAttr(f"{cube}.translate", 3.0, 0.0, 0.0)
    cmds.setAttr(f"{ball}.translate", -3.0, 0.0, 0.0)
    return {lambert: (plane_node, [0, 1]), phong: (plane_node, [2, 3]), blinn: (cube, None), standard: (ball, None)}


SHADER_ATTRS = ("color", "transparency", "ambientColor", "incandescence", "diffuse", "specularColor", "cosinePower", "reflectivity")


def shader_reading(shader: str) -> dict:
    held = {"type": cmds.nodeType(shader)}
    for attr in SHADER_ATTRS:
        if cmds.attributeQuery(attr, node=shader, exists=True):
            value      = cmds.getAttr(f"{shader}.{attr}")
            held[attr] = list(value[0]) if isinstance(value, list) else value
    return held


def materials_export(rec: Recorder) -> None:
    grad = rec.path("grad.png")
    Image.fromarray(mats.gradient()).save(grad)
    made = materials_scene(grad)
    rec.value("made",            {shader: [node, faces] for shader, (node, faces) in made.items()})
    rec.value("shaders",         {shader: shader_reading(shader) for shader in made})
    rec.value("gradPlace_wrapV", bool(cmds.getAttr("gradPlace.wrapV")))
    exported(rec.path("maya_embedded.fbx"), embed=True)
    exported(rec.path("maya_linked.fbx"), embed=False)


def materials_import(rec: Recorder) -> None:
    nodes = mats.rig()
    held  = mats.meshes(nodes)
    given = mats.materials(held)
    ours  = rec.path("ours.fbx")
    save_fbx(ours, nodes, meshes=held, uvs=mats.plane_uvs(held[0]), materials=given)
    imported(ours)
    readings = {}
    for material in given:
        shaders = cmds.ls(material.name, type=material.kind)
        entry   = {"shader_count": len(shaders)}
        if shaders:
            shader = shaders[0]
            entry.update(shader_reading(shader))
            files          = cmds.listConnections(f"{shader}.color", type="file") or []
            entry["files"] = len(files)
            if files:
                with Image.open(cmds.getAttr(f"{files[0]}.fileTextureName")) as image:
                    rec.array(f"texture/{material.name}", np.asarray(image.convert("RGB")))
                places          = cmds.listConnections(f"{files[0]}.uvCoord", type="place2dTexture") or []
                entry["places"] = len(places)
                if places:
                    entry["wraps"] = [bool(cmds.getAttr(f"{places[0]}.wrapU")), bool(cmds.getAttr(f"{places[0]}.wrapV"))]
            groups           = cmds.listConnections(f"{shader}.outColor", type="shadingEngine") or []
            entry["groups"]  = len(groups)
            entry["members"] = cmds.ls(cmds.sets(groups[0], q=True) or [], flatten=True, long=True) if groups else []
        readings[material.name] = entry
    rec.value("materials", readings)


def morphs_scene() -> dict:
    """the probe's scene: a 2 by 2 plane under blendShape faceShapes with
    smile (points 0 and 4 lifted 0.5 and 0.25; an in-between at 0.5 lifting
    points 0 and 8 by 0.2 and 0.1; weight 0.3) and frown (point 2 down 0.4;
    keyed 0 at frame 1, 0.5 at 3, 1 at 5); a cube under cubeShapes with
    puff (points 0 to 3 out 0.3; weight 0)"""
    cmds.file(new=True, force=True)
    cmds.currentUnit(time="ntsc", linear="cm", angle="deg")
    plane_node, _ = cmds.polyPlane(name="plane", width=2.0, height=2.0, subdivisionsX=2, subdivisionsY=2)
    smile = cmds.duplicate(plane_node, name="smile")[0]
    cmds.move(0.0, 0.5, 0.0, f"{smile}.vtx[0]", relative=True)
    cmds.move(0.0, 0.25, 0.0, f"{smile}.vtx[4]", relative=True)
    half = cmds.duplicate(plane_node, name="smileHalf")[0]
    cmds.move(0.0, 0.2, 0.0, f"{half}.vtx[0]", relative=True)
    cmds.move(0.0, 0.1, 0.0, f"{half}.vtx[8]", relative=True)
    frown = cmds.duplicate(plane_node, name="frown")[0]
    cmds.move(0.0, -0.4, 0.0, f"{frown}.vtx[2]", relative=True)
    blend = cmds.blendShape(smile, frown, plane_node, name="faceShapes", frontOfChain=True)[0]
    cmds.blendShape(blend, edit=True, inBetween=True, target=(plane_node, 0, half, 0.5))
    cmds.setAttr(f"{blend}.smile", 0.3)
    for frame, value in ((1, 0.0), (3, 0.5), (5, 1.0)):
        cmds.setKeyframe(f"{blend}.frown", time=frame, value=value)
    cmds.playbackOptions(minTime=1, maxTime=5, animationStartTime=1, animationEndTime=5)
    cmds.currentTime(1)
    cube, _ = cmds.polyCube(name="cube")
    puff = cmds.duplicate(cube, name="puff")[0]
    cmds.move(0.0, 0.0, 0.3, f"{puff}.vtx[0:3]", relative=True)
    cmds.blendShape(puff, cube, name="cubeShapes")
    cmds.setAttr(f"{cube}.translate", 3.0, 0.0, 0.0)
    cmds.hide([smile, half, frown, puff])
    return {"plane": plane_node, "cube": cube}


def morphs_export(rec: Recorder) -> None:
    morphs_scene()
    exported(rec.path("maya.fbx"), shapes=True)
    rec.value("frown_at_5", cmds.getAttr("faceShapes.frown", time=5))


def morphs_import(rec: Recorder) -> None:
    nodes = mats.rig()
    held  = morphs.meshes(nodes)
    given = morphs.face_morphs(held)
    ours  = rec.path("ours.fbx")
    save_fbx(ours, nodes, meshes=held, morphs=given)
    imported(ours)
    deformers = {}
    for deformer in ("faceShapes", "blendShape"):
        found = cmds.ls(deformer, type="blendShape")
        entry = {"nodes": found}
        if found:
            entry["alias"]   = cmds.aliasAttr(deformer, q=True) or []
            entry["weights"] = list(cmds.blendShape(deformer, q=True, weight=True) or [])
        deformers[deformer] = entry
    rec.value("deformers", deformers)
    rec.value("items",     sorted(cmds.getAttr("faceShapes.inputTarget[0].inputTargetGroup[0].inputTargetItem", multiIndices=True)))
    rec.value("extra",     cmds.ls("extra", type="blendShape"))
    for node, mesh in zip(nodes, held):
        rec.array(f"points/{node.name}", [cmds.pointPosition(f"{node.name}.vtx[{i}]", local=True) for i in range(mesh.point_count)])


def colors_export(rec: Recorder) -> None:
    cmds.file(new=True, force=True)
    transform = cmds.polyPlane(w=2.0, h=2.0, sx=2, sy=2, name="grid", constructionHistory=False)[0]
    shape     = cmds.listRelatives(transform, shapes=True)[0]
    cmds.polyColorSet(shape, create=True, colorSet="colorSet1", representation="RGBA")
    cmds.polyColorSet(shape, currentColorSet=True, colorSet="colorSet1")
    for v in range(9):
        cmds.polyColorPerVertex(f"{shape}.vtx[{v}]", rgb=(v / 8.0, 1.0 - v / 8.0, 0.25), alpha=0.5 + v / 16.0)
    sets, held = maya_colors(shape)
    rec.value("shared/sets", sets)
    rec.array("shared/colors", held)
    exported(rec.path("shared.fbx"), at=START)

    cmds.file(new=True, force=True)
    transform = cmds.polyPlane(w=2.0, h=2.0, sx=2, sy=2, name="rgb", constructionHistory=False)[0]
    shape     = cmds.listRelatives(transform, shapes=True)[0]
    cmds.polyColorSet(shape, create=True, colorSet="tint", representation="RGB")
    cmds.polyColorSet(shape, currentColorSet=True, colorSet="tint")
    for v in range(9):
        cmds.polyColorPerVertex(f"{shape}.vtx[{v}]", rgb=(0.1 * v, 0.0, 1.0 - 0.1 * v))
    sets, held = maya_colors(shape)
    rec.value("rgb/sets", sets)
    rec.array("rgb/colors", held)
    exported(rec.path("rgb.fbx"), at=START)


def colors_import(rec: Recorder) -> None:
    for form in ("vertex", "corner", "palette"):
        ours = rec.path(f"ours_{form}.fbx")
        save_fbx(ours, meshes=colors.colored(form))
        imported(ours)
        shape = cmds.ls(type="mesh")[0]
        sets, held = maya_colors(shape)
        rec.value(f"{form}/sets", sets)
        rec.array(f"{form}/colors", held)


def pivots_export(rec: Recorder) -> None:
    """a transform with every pivot and offset, a joint and an unevenly
    scaled pivoted leaf under it, keyed over 3 frames (test_fbx_pivots)"""
    new_scene()
    grp = cmds.createNode("transform", name="grp")
    cmds.setAttr(grp + ".rotatePivot", 1.0, 2.0, 3.0)
    cmds.setAttr(grp + ".scalePivot", -1.0, 0.5, 2.0)
    cmds.setAttr(grp + ".rotatePivotTranslate", 0.5, 0.0, 0.0)
    cmds.setAttr(grp + ".scalePivotTranslate", 0.0, 0.25, 0.0)
    cmds.setAttr(grp + ".rotate", 10.0, 20.0, 30.0)
    cmds.setAttr(grp + ".rotateOrder", 2)
    cmds.setAttr(grp + ".rotateAxis", 5.0, 0.0, 0.0)
    cmds.setAttr(grp + ".scale",      2.0, 2.0, 2.0)
    cmds.setAttr(grp + ".translate",  3.0, 4.0, 5.0)
    j = cmds.createNode("joint", name="j", parent=grp)
    cmds.setAttr(j + ".translate", 1.0, 2.0, 3.0)
    cmds.setAttr(j + ".jointOrient", 0.0, 15.0, 0.0)
    leaf = cmds.createNode("transform", name="leaf", parent=grp)
    cmds.setAttr(leaf + ".scale",       1.0, 2.0,  3.0)
    cmds.setAttr(leaf + ".rotatePivot", 2.0, 0.0,  0.0)
    cmds.setAttr(leaf + ".scalePivot",  0.0, 0.0,  1.0)
    cmds.setAttr(leaf + ".rotate",      0.0, 45.0, 0.0)
    cmds.setAttr(leaf + ".translate",   0.0, 1.0,  0.0)
    for frame, (rz, tx, sz) in enumerate(((0.0, 3.0, 3.0), (45.0, 4.0, 4.0), (90.0, 5.0, 5.0))):
        cmds.setKeyframe(grp,  attribute="rotateZ",    time=frame, value=rz)
        cmds.setKeyframe(grp,  attribute="translateX", time=frame, value=tx)
        cmds.setKeyframe(leaf, attribute="scaleZ",     time=frame, value=sz)
    cmds.playbackOptions(minTime=0, maxTime=2, animationStartTime=0, animationEndTime=2)
    for name in ("grp", "j", "leaf"):
        rec.array(f"world/{name}", [world(name, f) for f in range(3)])
    exported(rec.path("maya.fbx"), at=START)


def takes_each(rec: Recorder) -> None:
    clips        = clip_list.oriented_takes()
    clips.active = "c"
    ours         = rec.path("ours.fbx")
    clips.save_fbx(ours)
    back = ClipList.load_fbx(ours)
    for index, clip in enumerate(back):
        frames = clip.start_frame + np.arange(clip.frame_count)
        imported(ours, take=index + 1, units=False)
        nodes = cmds.ls(type="transform", long=True)
        for node in nodes:
            name = node.rsplit("|", 1)[-1]
            rows = []
            for frame in frames:
                cmds.currentTime(int(frame))
                rows.append(cmds.xform(node, q=True, ws=True, m=True))
            rec.array(f"take{index}/world/{name}", np.array(rows).reshape(-1, 4, 4))


def short_names_joints(rec: Recorder) -> None:
    """Maya's own joints posed as POSES with inverseScale wired"""
    new_scene()
    rig = short_names._rig()
    for name, parent, _ in short_names.POSES:
        joint = cmds.createNode("joint", name=name, **({"parent": parent} if parent else {}))
        if parent:
            cmds.connectAttr(parent + ".scale", joint + ".inverseScale")
        node = rig[name]
        for attr in ("t", "r", "s", "sh", "jo", "ra"):
            cmds.setAttr(f"{joint}.{attr}", *getattr(node, attr).tolist())
        for attr in ("ro", "ssc", "v", "radi", "ds"):
            cmds.setAttr(f"{joint}.{attr}", getattr(node, attr))
    rec.value("short_names", {
        maya_long: cmds.attributeQuery(maya_long, node="knee", shortName=True)
        for _, _, maya_long in short_names.MAYA_NAMES + (("is", None, "inverseScale"),)
    })
    plugs = {"m": "m", "wm": "wm[0]", "pm": "pm[0]", "im": "im", "wim": "wim[0]", "pim": "pim[0]"}
    for name, _, _ in short_names.POSES:
        for short, _, _ in short_names.MAYA_NAMES:
            rec.array(f"plug/{name}/{short}", np.array(cmds.getAttr(f"{name}.{plugs.get(short, short)}"), dtype=float))


def _matrix_plugs(name: str) -> dict:
    return {short: np.array(cmds.getAttr(f"{name}.{short}" + ("[0]" if short != "m" else ""))).reshape(4, 4) for short in ("m", "wm", "pm", "wim")}


def _wiring(name: str) -> dict:
    parent = cmds.listRelatives(name, parent=True)
    return {
        "parent":      parent[0] if parent else None,
        "parent_type": cmds.nodeType(parent[0]) if parent else None,
        "is_plugs":    cmds.listConnections(name + ".is", source=True, destination=False, plugs=True),
        "is":          list(cmds.getAttr(name + ".is")[0]),
        "ssc":         bool(cmds.getAttr(name + ".ssc")),
    }


def inverse_scale(rec: Recorder) -> None:
    """Maya's wiring of inverseScale as joints are drawn, made and moved"""
    new_scene()
    cmds.createNode("transform", name="group")
    cmds.select("group")
    cmds.joint(name="hip")
    cmds.joint(name="knee")
    cmds.setAttr("group.s", 2.0, 3.0, 4.0)
    cmds.setAttr("hip.s", 5.0, 6.0, 7.0)
    rec.value("drawn", {name: _wiring(name) for name in ("hip", "knee")})
    for name in ("hip", "knee"):
        for short, matrix in _matrix_plugs(name).items():
            rec.array(f"drawn/{name}/{short}", matrix)

    cmds.createNode("joint", name="shin", parent="hip")
    rec.value("shin", _wiring("shin"))
    for short, matrix in _matrix_plugs("shin").items():
        rec.array(f"shin/{short}", matrix)

    cmds.parent("knee", "group")
    above = cmds.listRelatives("knee", parent=True)
    rec.value("moved", {
        "above_type":   cmds.nodeType(above[0]),
        "above_parent": cmds.listRelatives(above[0], parent=True),
        "knee":         _wiring("knee"),
    })
    rec.array("moved/knee/m", np.array(cmds.getAttr("knee.m")).reshape(4, 4))

    relative = {}
    for name, drawn, under in (("heel", "hip", "group"), ("toe", "hip", "shin"), ("stray", None, "shin")):
        cmds.select(clear=True)
        if drawn:
            cmds.select(drawn)
        cmds.joint(name=name)
        cmds.parent(name, under, relative=True)
        relative[name] = _wiring(name)
    rec.value("relative", relative)

    for name in ("ankle", "loose"):
        cmds.select(clear=True)
        cmds.joint(name=name)
    cmds.parent("loose", "ankle")
    rec.value("loose_under_ankle", _wiring("loose"))
    cmds.parent("loose", "hip")
    above = cmds.listRelatives("loose", parent=True)
    rec.value("loose_under_hip", {
        "above_type":   cmds.nodeType(above[0]),
        "above_parent": cmds.listRelatives(above[0], parent=True),
        "is_plugs":     cmds.listConnections("loose.is", source=True, destination=False),
    })

    cmds.select(clear=True)
    cmds.joint(name="elbow")
    cmds.joint(name="wrist", position=(1.0, 2.0, 3.0))
    cmds.setAttr("elbow.s", 5.0, 6.0, 7.0)
    cmds.createNode("transform", name="sleeve")
    cmds.setAttr("sleeve.s", 5.0, 6.0, 7.0)
    cmds.parent("wrist", "sleeve")
    rec.value("wrist", _wiring("wrist"))
    for attr in ("t", "r", "s", "jo"):
        rec.array(f"wrist/{attr}", triple("wrist", attr))
    for compensate in (True, False):
        cmds.setAttr("wrist.ssc", compensate)
        for short in ("m", "wm"):
            rec.array(f"wrist/{'ssc' if compensate else 'free'}/{short}", np.array(cmds.getAttr("wrist." + short + ("[0]" if short != "m" else ""))).reshape(4, 4))


def shear_parent(rec: Recorder) -> None:
    """cmds.parent of a transform under a transform, 12 random trials, seed 13 (test_shear_maya)"""
    rng = np.random.default_rng(13)
    for trial in range(12):
        new_scene()
        parent = TransformData(
            "p",
            uuid      = "00000000-0000-4000-8000-000000000001",
            scale     = rng.uniform(0.4, 2.2, 3),
            rotate    = rng.uniform(-170.0, 170.0, 3),
            translate = rng.uniform(-3.0, 3.0, 3),
            shear     = rng.uniform(-1.0, 1.0, 3) if trial % 2 else None,
        )
        child = TransformData(
            "c",
            uuid         = "00000000-0000-4000-8000-000000000002",
            scale        = rng.uniform(0.4, 2.2, 3),
            rotate       = rng.uniform(-170.0, 170.0, 3),
            translate    = rng.uniform(-3.0, 3.0, 3),
            rotate_order = int(rng.integers(0, 6)),
            rotate_axis  = rng.uniform(-30.0, 30.0, 3),
        )
        HierarchyData([parent, child])
        for name, node in (("p", parent), ("c", child)):
            made = cmds.createNode("transform", name=name)
            for attr in ("t", "r", "s", "sh", "ra"):
                cmds.setAttr(f"{made}.{attr}", *getattr(node, attr).tolist())
            cmds.setAttr(f"{made}.ro", node.rotate_order)
        cmds.parent("c", "p")
        rec.array(f"trial{trial}/s",  triple("p|c", "s"))
        rec.array(f"trial{trial}/sh", triple("p|c", "sh"))
        rec.array(f"trial{trial}/m",  np.array(cmds.getAttr("p|c.m"), dtype=float).reshape(4, 4))


def _bumped(shader_node: str, file_name: str, texture: str, interp: int, depth: float) -> str:
    """file -> bump2d (bumpInterp ``interp``: 0 bump, 1 tangent space normals,
    2 object space normals; bumpDepth ``depth``) -> the shader's normalCamera"""
    file_node = cmds.shadingNode("file", asTexture=True, name=file_name)
    place     = cmds.shadingNode("place2dTexture", asUtility=True, name=file_name + "Place")
    cmds.connectAttr(f"{place}.outUV", f"{file_node}.uvCoord")
    cmds.connectAttr(f"{place}.outUvFilterSize", f"{file_node}.uvFilterSize")
    cmds.setAttr(f"{file_node}.fileTextureName", texture.replace(os.sep, "/"), type="string")
    bump = cmds.shadingNode("bump2d", asUtility=True, name=file_name + "Bump")
    cmds.setAttr(f"{bump}.bumpInterp", interp)
    cmds.setAttr(f"{bump}.bumpDepth", depth)
    cmds.connectAttr(f"{file_node}.outAlpha", f"{bump}.bumpValue")
    cmds.connectAttr(f"{bump}.outNormal", f"{shader_node}.normalCamera")
    return bump


def normals_scene(bump: str, normal: str, obj: str) -> dict:
    """the probe's scene: a plane under lambert bumpLambert whose bump2d reads
    ``bump`` as a bump map (depth 0.7), a cube under phong normalPhong whose
    bump2d reads ``normal`` as tangent space normals (depth 1.0), a sphere
    under lambert objectLambert whose bump2d reads ``obj`` as object space
    normals (depth 0.5); {shader: (mesh, bumpInterp, bumpDepth, file)}"""
    cmds.file(new=True, force=True)
    cmds.currentUnit(time="ntsc", linear="cm", angle="deg")

    def shader(kind, name, **values):
        made  = cmds.shadingNode(kind, asShader=True, name=name)
        group = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=f"{name}SG")
        cmds.connectAttr(f"{made}.outColor", f"{group}.surfaceShader", force=True)
        for attr, value in values.items():
            if isinstance(value, (tuple, list)):
                cmds.setAttr(f"{made}.{attr}", *value, type="double3")
            else:
                cmds.setAttr(f"{made}.{attr}", value)
        return made, group

    plane, _ = cmds.polyPlane(name="plane", width=2.0, height=2.0, subdivisionsX=1, subdivisionsY=1)
    lam, lam_sg = shader("lambert", "bumpLambert", color=(0.6, 0.6, 0.6))
    _bumped(lam, "bumpFile", bump, 0, 0.7)
    cmds.sets(plane, edit=True, forceElement=lam_sg)
    cube, _ = cmds.polyCube(name="cube")
    pho, pho_sg = shader("phong", "normalPhong", color=(0.9, 0.2, 0.1))
    _bumped(pho, "normalFile", normal, 1, 1.0)
    cmds.sets(cube, edit=True, forceElement=pho_sg)
    ball, _ = cmds.polySphere(name="ball", subdivisionsX=8, subdivisionsY=6)
    lam2, lam2_sg = shader("lambert", "objectLambert", color=(0.2, 0.2, 0.9))
    _bumped(lam2, "objectFile", obj, 2, 0.5)
    cmds.sets(ball, edit=True, forceElement=lam2_sg)
    cmds.setAttr(f"{cube}.translate", 3.0, 0.0, 0.0)
    cmds.setAttr(f"{ball}.translate", -3.0, 0.0, 0.0)
    return {lam: (plane, 0, 0.7, bump), pho: (cube, 1, 1.0, normal), lam2: (ball, 2, 0.5, obj)}


def normals_export(rec: Recorder) -> None:
    """Maya's export of a bump map, a tangent space normal map and object
    space normals through bump2d, linked and embedded"""
    bump, normal, obj = rec.path("bump.png"), rec.path("normal.png"), rec.path("object.png")
    Image.fromarray(mats.bump_map()).save(bump)
    Image.fromarray(mats.normal_map()).save(normal)
    Image.fromarray(mats.gradient()).save(obj)
    made = normals_scene(bump, normal, obj)
    rec.value("made", {shader: [node, interp, depth, os.path.basename(path)] for shader, (node, interp, depth, path) in made.items()})
    exported(rec.path("maya_linked.fbx"), embed=False)
    exported(rec.path("maya_embedded.fbx"), embed=True)


def normals_import(rec: Recorder) -> None:
    """Maya's import of our normal maps: the bump2d each shader gets"""
    nodes = mats.rig()
    held  = mats.meshes(nodes)
    given = mats.normal_materials(held)
    ours  = rec.path("ours.fbx")
    save_fbx(ours, nodes, meshes=held, uvs=UVList([mats.plane_uvs(held[0]), mats.plane_uvs(held[1])]), materials=given)
    imported(ours)
    readings = {}
    for material in given:
        shaders = cmds.ls(material.name, type=material.kind)
        entry   = {"shader_count": len(shaders)}
        if shaders:
            shader         = shaders[0]
            bumps          = cmds.listConnections(f"{shader}.normalCamera", source=True, destination=False, type="bump2d") or []
            entry["bumps"] = len(bumps)
            if bumps:
                entry["bumpInterp"] = cmds.getAttr(f"{bumps[0]}.bumpInterp")
                entry["bumpDepth"]  = cmds.getAttr(f"{bumps[0]}.bumpDepth")
                files               = cmds.listConnections(f"{bumps[0]}.bumpValue", source=True, destination=False, type="file") or []
                entry["files"]      = len(files)
                if files:
                    with Image.open(cmds.getAttr(f"{files[0]}.fileTextureName")) as image:
                        rec.array(f"normal_map/{material.name}", np.asarray(image.convert("RGB")))
            entry["color_files"] = len(cmds.listConnections(f"{shader}.color", type="file") or [])
        readings[material.name] = entry
    rec.value("materials", readings)


def default_material(rec: Recorder) -> None:
    """Maya's default material: a cube left on initialShadingGroup exported
    (its surface shader written bare), and our meshes with no material
    imported, the shading group each lands on"""
    cmds.file(new=True, force=True)
    cmds.currentUnit(time="ntsc", linear="cm", angle="deg")
    cube, _ = cmds.polyCube(name="cube")
    rec.value("default_shader", cmds.listConnections("initialShadingGroup.surfaceShader") or [])
    rec.value("cube_sets", cmds.listSets(object=cmds.listRelatives(cube, shapes=True)[0], type=1) or [])
    exported(rec.path("maya_cube.fbx"))
    nodes = mats.rig()
    held  = mats.meshes(nodes)
    ours  = rec.path("ours.fbx")
    save_fbx(ours, nodes, meshes=held)
    imported(ours)
    readings = {}
    for node in ("plane", "box"):
        shape          = cmds.listRelatives(node, shapes=True)[0]
        groups         = cmds.listSets(object=shape, type=1) or []
        readings[node] = {"groups": groups, "shaders": {g: (cmds.listConnections(f"{g}.surfaceShader") or []) for g in groups}}
    rec.value("imported", readings)
    # Maya's own export of the cube, imported back: the shading group and shader it gets
    imported(rec.path("maya_cube.fbx"))
    shape  = cmds.listRelatives("cube", shapes=True)[0]
    groups = cmds.listSets(object=shape, type=1) or []
    rec.value("reimported", {"groups": groups, "shaders": {g: (cmds.listConnections(f"{g}.surfaceShader") or []) for g in groups},
                              "materials": cmds.ls(materials=True)})


DERIVED_MESHES = (
    # name, kind, primitive, jitter, polySoftEdge angle (None: the edges Maya makes), the primitive's arguments
    ("softSheet", "soft", "polyPlane", 0.12, 180.0, {"subdivisionsX": 5, "subdivisionsY": 4}),
    ("softBall", "soft", "polySphere", 0.12, 180.0, {"subdivisionsX": 12, "subdivisionsY": 8}),
    ("softCan", "soft", "polyCylinder", 0.12, 180.0, {"subdivisionsX": 9, "subdivisionsY": 3, "subdivisionsCaps": 0}),
    ("softRing", "soft", "polyTorus", 0.12, 180.0, {"subdivisionsX": 10, "subdivisionsY": 6}),
    ("softCube", "soft", "polyCube", 0.12, 180.0, {}),
    ("cubeDefault", "hard", "polyCube", 0.0, None, {}),
    ("cubeJittered", "hard", "polyCube", 0.12, None, {}),
    ("canDefault", "hard", "polyCylinder", 0.0, None, {"subdivisionsX": 9, "subdivisionsY": 3, "subdivisionsCaps": 1}),
    ("canJittered", "hard", "polyCylinder", 0.12, None, {"subdivisionsX": 9, "subdivisionsY": 3, "subdivisionsCaps": 1}),
    ("coneJittered", "hard", "polyCone", 0.12, None, {"subdivisionsX": 8, "subdivisionsY": 2}),
    ("ballDefault", "hard", "polySphere", 0.0, None, {"subdivisionsX": 12, "subdivisionsY": 8}),
    ("ballSoft30", "hard", "polySphere", 0.12, 30.0, {"subdivisionsX": 12, "subdivisionsY": 8}),
    ("ringSoft45", "hard", "polyTorus", 0.12, 45.0, {"subdivisionsX": 10, "subdivisionsY": 6}),
    ("sheetSoft20", "hard", "polyPlane", 0.12, 20.0, {"subdivisionsX": 5, "subdivisionsY": 4}),
    ("lockedCube", "locked", "polyCube", 0.12, 180.0, {}),
)


def mesh_normals(name: str) -> dict:
    """the mesh as Maya holds it: its points, face counts and corner
    vertices, the normal of every face corner in face order with its id and
    its lock, and every edge's two vertices and whether it is smooth"""
    shape = om.MFnMesh(dag(name))
    sizes, ids = shape.getVertices()
    faces      = range(shape.numPolygons)
    normal_ids = [int(i) for face in faces for i in shape.getFaceNormalIds(face)]
    return {
        "points":     np.array([tuple(p)[:3] for p in shape.getPoints()]),
        "counts":     np.array(sizes, dtype=np.int32),
        "indices":    np.array(ids, dtype=np.int32),
        "normals":    np.array([tuple(n) for face in faces for n in shape.getFaceVertexNormals(face)], dtype=np.float64),
        "normal_ids": np.array(normal_ids, dtype=np.int32),
        "locked":     np.array([bool(shape.isNormalLocked(i)) for i in normal_ids]),
        "edges":      np.array([tuple(shape.getEdgeVertices(e)) for e in range(shape.numEdges)], dtype=np.int32),
        "smooth":     np.array([bool(shape.isEdgeSmooth(e)) for e in range(shape.numEdges)]),
    }


def derived_normals(rec: Recorder) -> None:
    """the normals Maya derives where none is authored: meshes with every
    edge soft, meshes with hard edges (the primitives with the edges Maya
    makes them with among them: a sphere's are all hard), and a cube with
    one vertex's normal locked; most are jittered so their quads bend. Each
    as Maya holds it, and all of them as Maya exports them"""
    new_scene()
    rng  = np.random.default_rng(11)
    info = {}
    for name, kind, primitive, jitter, soften, arguments in DERIVED_MESHES:
        node   = getattr(cmds, primitive)(name=name, constructionHistory=False, **arguments)[0]
        shape  = om.MFnMesh(dag(node))
        points = np.array([tuple(p)[:3] for p in shape.getPoints()])
        if jitter:
            points = points + rng.normal(scale=jitter, size=points.shape)
            shape.setPoints(om.MPointArray([om.MPoint(*p) for p in points]))
        if soften is not None:
            cmds.polySoftEdge(node, angle=soften, constructionHistory=False)
        if kind == "locked":
            cmds.polyNormalPerVertex(f"{node}.vtx[0]", xyz=(0.0, 1.0, 0.0))
        info[node] = {"kind": kind, "jittered": bool(jitter)}
    for name, entry in info.items():
        held = mesh_normals(name)
        for key, value in held.items():
            rec.array(f"{name}/{key}", value)
        entry["hard_edges"]     = int((~held["smooth"]).sum())
        entry["locked_corners"] = int(held["locked"].sum())
        entry["normals"]        = int(np.unique(held["normal_ids"]).size)
    rec.value("meshes", list(info))
    rec.value("info", info)
    exported(rec.path("maya.fbx"))


def _api_curve(name: str, points, knots, degree: int, form) -> str:
    """a curve built through the API, which takes a periodic or a closed
    form with the knots given"""
    parent = om.MFnTransform().create()
    om.MFnDependencyNode(parent).setName(name)
    cvs = [om.MPoint(*(float(c) for c in point)) for point in points]
    om.MFnNurbsCurve().create(cvs, [float(k) for k in knots], degree, form, False, False, parent)
    return name


def _under(parent: str, child: str) -> None:
    """``child``'s shape moved under ``parent``, beside its own"""
    cmds.parent(cmds.listRelatives(child, shapes=True, fullPath=True)[0], parent, relative=True, shape=True)
    cmds.delete(child)


def _cvs(grid) -> list:
    return [tuple(float(c) for c in point) for point in np.asarray(grid).reshape(-1, 3)]


def nurbs_readings(rec: Recorder, names) -> None:
    """what Maya holds under each transform of ``names``: its translate and,
    for each shape in Maya's order, the degree, the form, the knots and the
    control points, in object space"""
    for name in names:
        told = []
        for at, shape in enumerate(cmds.listRelatives(name, shapes=True, noIntermediate=True, fullPath=True) or []):
            chosen = om.MSelectionList()
            chosen.add(shape)
            path = chosen.getDagPath(0)
            kind = cmds.nodeType(shape)
            key  = f"{name}/{at}"
            if kind == "nurbsCurve":
                curve = om.MFnNurbsCurve(path)
                told.append({"type": kind, "degree": curve.degree, "form": curve.form})
                rec.array(f"{key}/knots", list(curve.knots()))
                rec.array(f"{key}/cvs", [tuple(p)[:3] for p in curve.cvPositions(om.MSpace.kObject)])
            elif kind == "nurbsSurface":
                surface = om.MFnNurbsSurface(path)
                told.append({
                    "type":     kind,
                    "degree_u": surface.degreeInU,
                    "degree_v": surface.degreeInV,
                    "form_u":   surface.formInU,
                    "form_v":   surface.formInV,
                    "count_u":  surface.numCVsInU,
                    "count_v":  surface.numCVsInV,
                })
                rec.array(f"{key}/knots_u", list(surface.knotsInU()))
                rec.array(f"{key}/knots_v", list(surface.knotsInV()))
                rec.array(f"{key}/cvs",     [tuple(p)[:3] for p in surface.cvPositions(om.MSpace.kObject)])
            else:
                told.append({"type": kind})
        rec.value(name, {"shapes": told, "translate": triple(name, "translate")})


def nurbs_scene() -> list:
    """a curve and a surface of every kind Maya's exporter writes, on the
    test module's points; the names of their transforms"""
    new_scene()
    loop  = list(nurbs.ring())
    shut  = list(nurbs.wave(4))
    forms = om.MFnNurbsCurve
    cmds.curve(n="open3", d=3, p=_cvs(nurbs.wave(6)), k=[0, 0, 0, 1, 2, 3, 3, 3])
    cmds.curve(n="knots3", d=3, p=_cvs(nurbs.wave(6, 1.0)), k=[0, 0, 0, 1, 2.5, 3, 3, 3])
    cmds.setAttr("knots3.translate", 1.0, 2.0, 3.0)
    # Maya's own count of a periodic curve: its first points again at the end
    _api_curve("loop3",     loop + loop[:3], range(-2, 9),          3, forms.kPeriodic)
    _api_curve("loopKnots", loop + loop[:3], nurbs.LOOP_KNOTS,      3, forms.kPeriodic)
    _api_curve("closed3",   shut + shut[:1], [0, 0, 0, 1, 2, 2, 2], 3, forms.kClosed)
    cmds.curve(n="degree2",   d=2, p=_cvs(nurbs.wave(5)),      k=[0, 0, 1, 2, 3, 3])
    cmds.curve(n="line",      d=1, p=_cvs(nurbs.wave(4, 2.0)), k=[0, 1, 2, 3])
    cmds.curve(n="lineKnots", d=1, p=_cvs(nurbs.wave(4)),      k=[0, 1, 3, 4])
    _api_curve("loop1", loop + loop[:1], range(7), 1, forms.kPeriodic)
    cmds.curve(n="lines", d=1, p=_cvs(nurbs.wave(2)), k=[0, 1])
    _under("lines", cmds.curve(n="linesB", d=1, p=_cvs(nurbs.wave(3, 1.0)), k=[0, 1, 2]))
    cmds.surface(n="patch", du=3, dv=3, ku=[0, 0, 0, 1, 1, 1], kv=[0, 0, 0, 1, 2, 2, 2], p=_cvs(nurbs.sheet(4, 5)))
    cmds.surface(
        n="patchKnots", du=2, dv=3, ku=[0, 0, 0.7, 2, 2], kv=[0, 0, 0, 1.5, 2, 2, 2], p=_cvs(nurbs.sheet(4, 5) + 1.0)
    )
    cmds.cylinder(n="tube", s=6, nsp=2, ch=False, ax=(0, 1, 0))
    return [
        "open3", "knots3", "loop3", "loopKnots", "closed3", "degree2", "line", "lineKnots", "loop1", "lines", "patch",
        "patchKnots", "tube",
    ]


def nurbs_export(rec: Recorder) -> None:
    names = nurbs_scene()
    rec.value("nodes", names)
    nurbs_readings(rec, names)
    exported(rec.path("maya.fbx"))
    # the shapes the fork refuses, each in a file beside one it reads
    plain = {"d": 3, "p": _cvs(nurbs.wave(6)), "k": [0, 0, 0, 1, 2, 3, 3, 3]}
    new_scene()
    cmds.curve(n="rational", d=2, pw=[(0, 0, 0, 1), (1, 1, 0, 0.5), (2, 0, 0, 1)], k=[0, 0, 1, 1])
    cmds.surface(n="patch", du=3, dv=3, ku=[0, 0, 0, 1, 1, 1], kv=[0, 0, 0, 1, 2, 2, 2], p=_cvs(nurbs.sheet(4, 5)))
    exported(rec.path("maya_rational.fbx"))
    new_scene()
    parent = om.MFnTransform().create()
    om.MFnDependencyNode(parent).setName("rationalPatch")
    cvs  = [om.MPoint(u, u * v * 0.25, v, 2.0 if (u, v) == (1, 1) else 1.0) for u in range(4) for v in range(4)]
    side = om.MFnNurbsSurface.kOpen
    om.MFnNurbsSurface().create(cvs, [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], 3, 3, side, side, True, parent)
    cmds.curve(n="plain", **plain)
    exported(rec.path("maya_rational_surface.fbx"))
    new_scene()
    trimmed = cmds.nurbsPlane(n="trimmed", u=2, v=2, ch=False, ax=(0, 1, 0))[0]
    hole    = cmds.circle(n="trimRing", r=0.2, nr=(0, 1, 0), ch=False)[0]
    cmds.move(0, 1, 0, hole)
    cmds.projectCurve(hole, trimmed, d=(0, 1, 0), ch=False)
    cmds.trim(trimmed, lu=0.9, lv=0.9, ch=False)
    cmds.delete(hole)
    cmds.curve(n="plain", **plain)
    exported(rec.path("maya_trimmed.fbx"))


def nurbs_import(rec: Recorder) -> None:
    nodes = nurbs.rig()
    ours  = rec.path("ours.fbx")
    save_fbx(ours, nodes, curves=nurbs.curves(nodes), surfaces=nurbs.surfaces(nodes))
    imported(ours)
    nurbs_readings(rec, list(nurbs.NODES) + ["curve", "surface"])


CASES = {
    "clip_import": clip_import,
    "visibility_locator": visibility_locator,
    "rrss": rrss,
    "takes": takes,
    "skeleton_import": skeleton_import,
    "export_reads_back": export_reads_back,
    "skin_geo": skin_geo,
    "skin_placed": skin_placed,
    "two_meshes": two_meshes,
    "uvs_import": uvs_import,
    "uvs_export": uvs_export,
    "uvs_none": uvs_none,
    "zup": zup,
    "zup_skinned": zup_skinned,
    "views_export": views_export,
    "views_import": views_import,
    "other_writer": other_writer,
    "materials_export": materials_export,
    "materials_import": materials_import,
    "normals_export": normals_export,
    "normals_import": normals_import,
    "default_material": default_material,
    "derived_normals": derived_normals,
    "morphs_export": morphs_export,
    "morphs_import": morphs_import,
    "colors_export": colors_export,
    "colors_import": colors_import,
    "pivots_export": pivots_export,
    "takes_each": takes_each,
    "short_names_joints": short_names_joints,
    "inverse_scale": inverse_scale,
    "shear_parent": shear_parent,
    "nurbs_export": nurbs_export,
    "nurbs_import": nurbs_import,
}


def main(names) -> None:
    chosen = names or list(CASES)
    for name in chosen:
        if name not in CASES:
            raise SystemExit(f"no case {name!r}; the cases are {list(CASES)}")
    os.makedirs(OUT, exist_ok=True)
    for name in chosen:
        rec = Recorder(name)
        CASES[name](rec)
        rec.save()
    manifest = os.path.join(OUT, "manifest.json")
    held     = {}
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as f:
            held = json.load(f)
    held.update({"maya": cmds.about(version=True), "fbx_plugin": cmds.pluginInfo("fbxmaya", q=True, version=True)})
    held.setdefault("cases", {})
    for name in chosen:
        held["cases"][name] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(manifest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(held, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"recorded {len(chosen)} cases with Maya {held['maya']} into {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
    maya.standalone.uninitialize()
