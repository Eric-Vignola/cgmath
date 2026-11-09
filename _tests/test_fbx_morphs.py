"""Blend shapes through load_fbx and save_fbx: Maya's conventions, measured.

Maya's exporter (2025) writes a blendShape node as one FbxBlendShape of its
name on the mesh, each target as a channel named "deformer.target" (its
import strips the prefix) whose DeformPercent is the weight times 100, each
target shape over every point of the mesh, absolute, with normals, an
in-between as a second shape of the channel at its weight times 100 (the
full weights); a keyed weight as a curve on DeformPercent. The fork reads
and writes the same, less the normals. The SDK tests need the Autodesk SDK,
the Maya ones a Maya with its fbxmaya plug-in.
"""

import os
import unittest
import warnings

import numpy as np
from cgmath._tests.test_fbx_materials import box, plane, rig
from cgmath._tests.test_skeleton_fbx import _Files, _InMaya
from cgmath.formats import load_fbx, save_fbx
from cgmath.formats._fbx_io import load_sdk, open_fbx, typed
from cgmath.geometry import MeshList, MorphData, MorphList
from cgmath.hierarchy import HierarchyData

FBX = load_sdk()


def meshes(nodes) -> MeshList:
    return MeshList([plane(nodes["plane"].uuid), box(nodes["box"].uuid)])


def face_morphs(held) -> MorphList:
    """on the plane under "faceShapes": smile (weight 0.3) with an
    in-between at 0.5, which a file holds first, frown (0.75) and wide (1);
    on the box: puff (0) under no deformer"""
    face, cube = held[0].uuid, held[1].uuid
    return MorphList([
        MorphData("smile", [[0.0, 0.2, 0.0], [0.0, 0.1, 0.0]], [0, 8], mesh_uuid=face, weight=0.3, at=0.5, deformer="faceShapes"),
        MorphData("smile", [[0.0, 0.5, 0.0], [0.0, 0.25, 0.0]], [0, 4], mesh_uuid=face, weight=0.3, deformer="faceShapes"),
        MorphData("frown", [[0.0, -0.4, 0.0]], [2], mesh_uuid=face, weight=0.75, deformer="faceShapes"),
        MorphData("wide", [[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], [1, 7], mesh_uuid=face, weight=1.0, deformer="faceShapes"),
        MorphData("puff", [[0.0, 0.0, 0.3]] * 4, [0, 1, 2, 3], mesh_uuid=cube, weight=0.0),
    ])


def same_morphs(test, got, want, got_meshes, want_meshes, exact: bool = False) -> None:
    """``got``, a MorphList read back, holds ``want``'s targets in order:
    name, deformer ("blendShape" for None), weight, ``at`` and indices
    equal, offsets within 1e-12 (equal when ``exact``), each on the mesh of
    ``got_meshes`` at the place its original has in ``want_meshes``"""
    test.assertEqual(type(got), MorphList)
    test.assertEqual([m.name for m in got], [m.name for m in want])
    for a, b in zip(got, want):
        test.assertEqual((a.deformer, a.weight, a.at), (b.deformer or "blendShape", b.weight, b.at), a.name)
        np.testing.assert_array_equal(a.indices, b.indices, err_msg=a.name)
        if exact:
            np.testing.assert_array_equal(a.offsets, b.offsets, err_msg=a.name)
        else:
            np.testing.assert_allclose(a.offsets, b.offsets, rtol=0, atol=1e-12, err_msg=a.name)
        place = [mesh.uuid for mesh in want_meshes].index(b.mesh_uuid)
        test.assertEqual(a.mesh_uuid, got_meshes[place].uuid, a.name)


def fbx_morphs(path: str) -> dict:
    """{node name: [(deformer name, [(channel name, DeformPercent, full
    weights, [(shape name, point count, moved points)])])]} of a file's
    mesh nodes, through the sdk (which hands every shape a normal element on
    read, written or not)"""
    out = {}
    with open_fbx(path, FBX) as scene:
        kind  = FBX.FbxDeformer.EDeformerType.eBlendShape
        stack = [scene.GetRootNode()]
        while stack:
            node = stack.pop()
            for i in range(node.GetChildCount()):
                stack.append(node.GetChild(i))
            attribute = node.GetNodeAttribute()
            if attribute is None or attribute.GetAttributeType() != FBX.FbxNodeAttribute.EType.eMesh:
                continue
            mesh  = typed(attribute, FBX.FbxMesh, FBX)
            base  = np.array([list(mesh.GetControlPointAt(k))[:3] for k in range(mesh.GetControlPointsCount())])
            found = []
            for d in range(mesh.GetDeformerCount(kind)):
                deformer = typed(mesh.GetDeformer(d, kind), FBX.FbxBlendShape, FBX)
                channels = []
                for c in range(deformer.GetBlendShapeChannelCount()):
                    channel = deformer.GetBlendShapeChannel(c)
                    shapes  = []
                    for s in range(channel.GetTargetShapeCount()):
                        shape  = channel.GetTargetShape(s)
                        points = np.array([list(shape.GetControlPointAt(k))[:3] for k in range(shape.GetControlPointsCount())])
                        moved  = np.flatnonzero(np.any(points != base, axis=1)).tolist() if len(points) == len(base) else None
                        shapes.append((str(shape.GetName()), shape.GetControlPointsCount(), moved))
                    channels.append((
                        str(channel.GetName()), float(channel.DeformPercent.Get()),
                        [float(w) for w in channel.GetTargetShapeFullWeights()], shapes,
                    ))
                found.append((str(deformer.GetName()), channels))
            out[str(node.GetName())] = found
    return out


def other_writer(folder: str, variant: str = "keyed") -> str:
    """an sdk file as another writer makes one: node quad, a 2 by 2 plane,
    with a deformer of no name holding channel "grin" (no prefix) at
    DeformPercent 40, keyed 0 to 100 over frames 1 to 5 at 24 fps, one
    target moving point 4; the variant "count" a channel "wrong" whose
    target holds 3 points; "zero" a channel "flat" whose target's full
    weight is 0; "metres" the file in metres, the target moving point 4 by
    0.01"""
    path    = os.path.join(folder, f"{variant}.fbx")
    manager = FBX.FbxManager.Create()
    manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
    try:
        scene = FBX.FbxScene.Create(manager, "")
        scene.GetGlobalSettings().SetTimeMode(FBX.FbxTime.EMode.eFrames24)
        if variant == "metres":
            scene.GetGlobalSettings().SetSystemUnit(FBX.FbxSystemUnit.m)
        node = FBX.FbxNode.Create(scene, "quad")
        scene.GetRootNode().AddChild(node)
        shape  = FBX.FbxMesh.Create(scene, "quadShape")
        points = [(x, 0.0, z) for z in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)]
        shape.InitControlPoints(len(points))
        for i, point in enumerate(points):
            shape.SetControlPointAt(FBX.FbxVector4(*point, 1.0), i)
        for face in ((0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)):
            shape.BeginPolygon(-1, -1, -1, False)
            for v in face:
                shape.AddPolygon(v)
            shape.EndPolygon()
        node.SetNodeAttribute(shape)

        def target(name, moved, lift, count=None):
            made = FBX.FbxShape.Create(scene, name)
            held = points if count is None else points[:count]
            made.InitControlPoints(len(held))
            for i, point in enumerate(held):
                point = (point[0], point[1] + (lift if i in moved else 0.0), point[2])
                made.SetControlPointAt(FBX.FbxVector4(*point, 1.0), i)
            return made

        deformer = FBX.FbxBlendShape.Create(scene, "")
        if variant == "count":
            channel = FBX.FbxBlendShapeChannel.Create(scene, "wrong")
            channel.AddTargetShape(target("wrong", [0], 0.5, count=3), 100.0)
        elif variant == "zero":
            channel = FBX.FbxBlendShapeChannel.Create(scene, "flat")
            channel.AddTargetShape(target("flat", [4], 0.5), 0.0)
        else:
            channel = FBX.FbxBlendShapeChannel.Create(scene, "grin")
            channel.DeformPercent.Set(40.0)
            channel.AddTargetShape(target("grin", [4], 0.01 if variant == "metres" else 0.5), 100.0)
        deformer.AddBlendShapeChannel(channel)
        shape.AddDeformer(deformer)

        if variant == "keyed":
            stack = FBX.FbxAnimStack.Create(scene, "take")
            layer = FBX.FbxAnimLayer.Create(scene, "BaseLayer")
            stack.AddMember(layer)
            times = []
            for frame in range(1, 6):
                when = FBX.FbxTime()
                when.SetFrame(frame, FBX.FbxTime.EMode.eFrames24)
                times.append(when)
            stack.SetLocalTimeSpan(FBX.FbxTimeSpan(times[0], times[-1]))
            channel.DeformPercent.GetCurveNode(layer, True)
            curve = typed(channel.DeformPercent.GetCurve(layer, True), FBX.FbxAnimCurve, FBX)
            curve.KeyModifyBegin()
            for when, value in zip(times, np.linspace(0.0, 100.0, 5)):
                index = curve.KeyAdd(when)[0]
                curve.KeySetValue(index, float(value))
                curve.KeySetInterpolation(index, FBX.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear)
            curve.KeyModifyEnd()

        exporter = FBX.FbxExporter.Create(manager, "")
        if not exporter.Initialize(path, -1, manager.GetIOSettings()):
            raise RuntimeError(f"cannot write {path}")
        exporter.Export(scene)
        exporter.Destroy()
    finally:
        manager.Destroy()
    return path


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMorphs(_Files):
    """blend shape targets on their meshes, saved and read back"""

    def test_round_trip_and_what_the_file_holds(self):
        """every target back on its mesh with its deformer, weight, in-between
        place and offsets; a second cycle bit for bit; the file as Maya
        writes one: a deformer a name, a channel "deformer.name" a target
        name with DeformPercent = weight x 100, its shapes by at with full
        weights at x 100, each over every point"""
        nodes = rig()
        held  = meshes(nodes)
        given = face_morphs(held)
        save_fbx(self.path("a.fbx"), nodes, meshes=held, morphs=given)
        back = load_fbx(self.path("a.fbx"))
        self.assertSameSkeleton(back.hierarchy, nodes)
        same_morphs(self, back.morphs, given, back.meshes, held)
        self.assertEqual([m.deformer for m in back.morphs], ["faceShapes"] * 4 + ["blendShape"])
        save_fbx(self.path("b.fbx"), *back)
        again = load_fbx(self.path("b.fbx"))
        same_morphs(self, again.morphs, back.morphs, again.meshes, back.meshes, exact=True)
        # the class routes read the nodes and the meshes alone
        self.assertEqual(HierarchyData.load_fbx(self.path("a.fbx")).name, nodes.name)

        plane_held = [
            ("faceShapes", [
                ("faceShapes.smile", 30.0, [50.0, 100.0], [("smile_50", 9, [0, 8]), ("smile", 9, [0, 4])]),
                ("faceShapes.frown", 75.0, [100.0], [("frown", 9, [2])]),
                ("faceShapes.wide", 100.0, [100.0], [("wide", 9, [1, 7])]),
            ]),
        ]
        found = fbx_morphs(self.path("a.fbx"))
        self.assertEqual(sorted(found),  ["box", "plane"])
        self.assertEqual(found["plane"], plane_held)
        self.assertEqual(found["box"],   [("blendShape", [("blendShape.puff", 0.0, [100.0], [("puff", 8, [0, 1, 2, 3])])])])
        # the full target listed ahead of its in-between: the file still holds them by at
        swapped = MorphList([given[1], given[0]] + list(given[2:]))
        save_fbx(self.path("c.fbx"), nodes, meshes=held, morphs=swapped)
        self.assertEqual(fbx_morphs(self.path("c.fbx"))["plane"], plane_held)
        third = load_fbx(self.path("c.fbx"))
        same_morphs(self, third.morphs, given, third.meshes, held)

    def test_another_writers_file(self):
        """a channel named without the deformer's prefix under a deformer of
        no name; a keyed weight read at the first frame with one warning; a
        target the sdk wrote with another point count moves nothing, one
        reached at 0 is refused; a file in metres gives offsets in cm"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            back = load_fbx(other_writer(self.folder))
        messages = [str(w.message) for w in caught if "weight is keyed" in str(w.message)]
        self.assertEqual(len(messages), 1)
        self.assertIn("blend shape target 'grin' of mesh 'quadShape': the weight is keyed in take 'take'", messages[0])
        self.assertEqual(len(back.morphs), 1)
        grin = back.morphs[0]
        self.assertEqual((grin.name, grin.deformer, grin.weight, grin.at, grin.mesh_uuid), ("grin", None, 0.0, 1.0, back.meshes[0].uuid))
        np.testing.assert_array_equal(grin.indices, [4])
        np.testing.assert_allclose(grin.offsets, [[0.0, 0.5, 0.0]], rtol=0, atol=1e-12)
        self.assertTrue(back.clip.active.take == "take")

        # the sdk writes a shape as deltas over the mesh's points and drops
        # one of another count (measured): it reads back moving nothing
        wrong = load_fbx(other_writer(self.folder, "count")).morphs[0]
        self.assertEqual((wrong.name, wrong.indices.tolist(), wrong.offsets.shape), ("wrong", [], (0, 3)))
        with self.assertRaisesRegex(ValueError, "mesh 'quadShape' blend shape target 'flat' is reached at weight 0.0, which no target is"):
            load_fbx(other_writer(self.folder, "zero"))
        metres = load_fbx(other_writer(self.folder, "metres"))
        np.testing.assert_allclose(metres.morphs[0].offsets,   [[0.0, 1.0, 0.0]],   rtol=0, atol=1e-9)
        np.testing.assert_allclose(metres.meshes[0].points[4], [0.0, 0.0, 0.0],     rtol=0, atol=1e-9)
        np.testing.assert_allclose(metres.meshes[0].points[8], [100.0, 0.0, 100.0], rtol=0, atol=1e-9)

    def test_refusals(self):
        nodes = rig()
        held  = meshes(nodes)
        face  = held[0].uuid

        def refused(message, given, kind=ValueError):
            with self.assertRaisesRegex(kind, message):
                save_fbx(self.path("refused.fbx"), nodes, meshes=held, morphs=given)
            self.assertFalse(os.path.exists(self.path("refused.fbx")))

        refused("blend shape target 'lost' names no mesh of meshes=: mesh_uuid None", MorphData("lost", [[0.0, 1.0, 0.0]], [0]))
        refused("blend shape target 'far' of mesh 'planeShape' channel 'indices' names points mesh 'planeShape' of 9 points does not hold",
                MorphData("far", [[0.0, 1.0, 0.0]], [9], mesh_uuid=face))
        refused("blend shape target 'odd' of mesh 'planeShape' holds 2 offsets for 1 indices",
                MorphData("odd", [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], [0], mesh_uuid=face))
        refused(r"mesh 'planeShape' holds blend shape deformers \['faceShapes', 'extra'\]; Maya's import deforms a mesh by its last",
                MorphList([MorphData("a", [[0.0, 1.0, 0.0]], [0], mesh_uuid=face, deformer="faceShapes"),
                           MorphData("b", [[0.0, 1.0, 0.0]], [0], mesh_uuid=face, deformer="extra")]))
        refused("blend shape target 'half' of mesh 'planeShape' under deformer 'blendShape' has 0 targets at 1 of 1",
                MorphData("half", [[0.0, 1.0, 0.0]], [0], mesh_uuid=face, at=0.5))
        refused("blend shape target 'twice' of mesh 'planeShape' under deformer 'face' has 2 targets at 1 of 2",
                MorphList([MorphData("twice", [[0.0, 1.0, 0.0]], [0], mesh_uuid=face, deformer="face"),
                           MorphData("twice", [[0.0, 2.0, 0.0]], [0], mesh_uuid=face, deformer="face")]))
        refused("blend shape target 'same' of mesh 'planeShape' under deformer 'blendShape' has two targets at 0.5",
                MorphList([MorphData("same", [[0.0, 1.0, 0.0]], [0], mesh_uuid=face),
                           MorphData("same", [[0.0, 0.5, 0.0]], [0], mesh_uuid=face, at=0.5),
                           MorphData("same", [[0.0, 0.6, 0.0]], [0], mesh_uuid=face, at=0.5)]))
        refused("blend shape target 'over' of mesh 'planeShape' under deformer 'blendShape' has a target at 1.5, above its full target at 1",
                MorphList([MorphData("over", [[0.0, 1.0, 0.0]], [0], mesh_uuid=face),
                           MorphData("over", [[0.0, 1.5, 0.0]], [0], mesh_uuid=face, at=1.5)]))
        refused(r"blend shape target 'split' of mesh 'planeShape' under deformer 'blendShape' holds weights \[0.3, 0.5\]; a channel holds one weight",
                MorphList([MorphData("split", [[0.0, 1.0, 0.0]], [0], mesh_uuid=face, weight=0.5),
                           MorphData("split", [[0.0, 0.5, 0.0]], [0], mesh_uuid=face, at=0.5, weight=0.3)]))
        refused("the morphs are a MorphList or a MorphData, not a list", [MorphData("x", [[0.0, 1.0, 0.0]], [0])], TypeError)
        with self.assertRaisesRegex(TypeError, "morphs are written on their mesh; missing: meshes="):
            save_fbx(self.path("refused.fbx"), nodes, morphs=MorphData("x", [[0.0, 1.0, 0.0]], [0], mesh_uuid=face))
        # a mesh with no target writes none, an empty list none
        save_fbx(self.path("none.fbx"), nodes, meshes=held, morphs=MorphList())
        self.assertIsNone(load_fbx(self.path("none.fbx")).morphs)
        self.assertEqual(fbx_morphs(self.path("none.fbx"))["plane"], [])


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMayaMorphs(_InMaya):
    """Maya 2025's export of its blend shapes read as Maya holds them, and
    Maya's import of ours"""

    def maya_scene(self) -> dict:
        """the probe's scene: a 2 by 2 plane under blendShape faceShapes with
        smile (points 0 and 4 lifted 0.5 and 0.25; an in-between at 0.5
        lifting points 0 and 8 by 0.2 and 0.1; weight 0.3) and frown (point
        2 down 0.4; keyed 0 at frame 1, 0.5 at 3, 1 at 5); a cube under
        cubeShapes with puff (points 0 to 3 out 0.3; weight 0);
        {(deformer, target, at): (mesh transform, {point: offset})}"""
        cmds = self.cmds
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
        return {
            ("faceShapes", "smile", 1.0): (plane_node, {0: (0.0, 0.5, 0.0), 4: (0.0, 0.25, 0.0)}),
            ("faceShapes", "smile", 0.5): (plane_node, {0: (0.0, 0.2, 0.0), 8: (0.0, 0.1, 0.0)}),
            ("faceShapes", "frown", 1.0): (plane_node, {2: (0.0, -0.4, 0.0)}),
            ("cubeShapes", "puff", 1.0): (cube, {i: (0.0, 0.0, 0.3) for i in range(4)}),
        }

    def exported_shapes(self, name: str) -> str:
        cmds, mel = self.cmds, self.mel
        mel.eval("FBXResetExport")
        mel.eval("FBXExportConvertUnitString cm")
        mel.eval("FBXExportUpAxis y")
        mel.eval("FBXExportShapes -v true")
        out = self.path(name).replace(os.sep, "/")
        mel.eval(f'FBXExport -f "{out}"')
        return out

    def test_mayas_export_reads_as_maya_holds_it(self):
        """every target with its deformer, name, weight (the keyed one at
        frame 1, with one warning), in-between place and the offsets of
        the points Maya moved, on its mesh"""
        cmds = self.cmds
        want = self.maya_scene()
        out  = self.exported_shapes("maya.fbx")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            back = load_fbx(out)
        messages = [str(w.message) for w in caught if "weight is keyed" in str(w.message)]
        self.assertEqual(len(messages), 1)
        self.assertIn("blend shape target 'frown'", messages[0])
        rig    = back.hierarchy if back.hierarchy is not None else back.clip.active
        meshes = {rig[mesh.parent_uuid].name: mesh for mesh in back.meshes}
        got    = {(m.deformer, m.name, m.at): m for m in back.morphs}
        self.assertEqual(sorted(got), sorted(want))
        for key, (node, offsets) in want.items():
            morph = got[key]
            with self.subTest(key):
                self.assertEqual(morph.mesh_uuid, meshes[node].uuid)
                np.testing.assert_array_equal(morph.indices, sorted(offsets))
                np.testing.assert_allclose(morph.offsets, [offsets[i] for i in sorted(offsets)], rtol=0, atol=1e-6)
        self.assertAlmostEqual(got[("faceShapes", "smile", 1.0)].weight, 0.3, places=6)
        self.assertAlmostEqual(got[("faceShapes", "smile", 0.5)].weight, 0.3, places=6)
        self.assertEqual(got[("faceShapes", "frown", 1.0)].weight, 0.0)  # frame 1's key
        self.assertEqual(got[("cubeShapes", "puff", 1.0)].weight,  0.0)
        self.assertEqual(cmds.getAttr("faceShapes.frown", time=5), 1.0)  # what the static read leaves

    def test_maya_imports_our_save_as_we_hold_it(self):
        """our targets, imported: blendShape nodes of our deformer names with
        our targets as aliases, our weights, the in-between at 0.5, and the
        meshes deformed as Maya blends them within 1e-6"""
        cmds  = self.cmds
        nodes = rig()
        held  = meshes(nodes)
        given = face_morphs(held)
        save_fbx(self.path("a.fbx"), nodes, meshes=held, morphs=given)
        self.imported(self.path("a.fbx"))
        for deformer, targets in (("faceShapes", ["smile", "frown", "wide"]), ("blendShape", ["puff"])):
            with self.subTest(deformer):
                self.assertEqual(cmds.ls(deformer, type="blendShape"), [deformer])
                alias = cmds.aliasAttr(deformer, q=True) or []
                self.assertEqual(alias[::2], targets)
                weights = [m.weight for m in given if (m.deformer or "blendShape") == deformer and m.at == 1.0]
                np.testing.assert_allclose(cmds.blendShape(deformer, q=True, weight=True), weights, rtol=0, atol=1e-6)
        items = cmds.getAttr("faceShapes.inputTarget[0].inputTargetGroup[0].inputTargetItem", multiIndices=True)
        self.assertEqual(sorted(items), [5500, 6000])  # the in-between at 0.5, the full target at 1
        self.assertEqual(cmds.ls("extra", type="blendShape"), [])
        # the deformed points: Maya blends a weight below an in-between's
        # place between the base and the in-between, in proportion
        expected = {mesh.uuid: np.asarray(mesh.points, dtype=np.float64).copy() for mesh in held}
        expected[held[0].uuid][[0, 8]] += 0.3 / 0.5 * np.array([[0.0, 0.2, 0.0], [0.0, 0.1, 0.0]])  # smile 0.3, in-between at 0.5
        expected[held[0].uuid][[2]]    += 0.75 * np.array([[0.0, -0.4, 0.0]])                       # frown 0.75
        expected[held[0].uuid][[1, 7]] += np.array([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]])             # wide 1
        for node, mesh in zip(nodes, held):
            with self.subTest(node.name):
                points = np.array([cmds.pointPosition(f"{node.name}.vtx[{i}]", local=True) for i in range(mesh.point_count)])
                np.testing.assert_allclose(points, expected[mesh.uuid], rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
