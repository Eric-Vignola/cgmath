"""NURBS curves and surfaces through load_fbx and save_fbx: Maya's conventions, measured.

Maya's exporter (2025) writes a curve of degree 2 or more as an
FbxNurbsCurve, its knots Maya's own with one more at each end, a periodic
curve as its unique control points, a closed one less its repeated last
point; a curve of degree 1 as an FbxLine, which holds no knots and no form
(a periodic one returns open, its first point repeated), the curves of
degree 1 under one transform as the lines of one FbxLine; a surface as an
FbxNurbsSurface, its points u fastest; a trimmed surface as an
FbxTrimNurbsSurface. A transform holding two curves of degree 2 or more
is written as an empty node. A rational shape holds a weight a control
point, the point's fourth value, the point itself as it stands: Maya's cvs
and its file are not multiplied by their weights. The fork reads and writes
the same, refusing a trimmed surface. The SDK tests need the Autodesk SDK;
the Maya ones read what Maya 2025 did, recorded once (maya_fixtures), and
need no Maya.
"""

import unittest

import numpy as np
from cgmath._tests.test_fbx_materials import plane
from cgmath._tests.test_skeleton_fbx import _Files, _Recorded
from cgmath.formats import load_fbx, save_fbx
from cgmath.formats._fbx_io import load_sdk, open_fbx, typed
from cgmath.geometry import MeshList, NurbsCurveData, NurbsCurveList, NurbsSurfaceData, NurbsSurfaceList
from cgmath.hierarchy import HierarchyData, TransformData

FBX = load_sdk()

# the transforms of the rig, one per shape but "lines", which holds two
NODES = (
    "open3", "knots3", "loop3", "loopKnots", "degree2", "degree1", "loop1", "line", "lines", "patch", "tube",
    "patchKnots", "arc", "edge", "bulge",
)
# a quarter circle of radius 1: its middle point's weight is the cosine of 45 degrees
ARC          = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
ARC_WEIGHTS  = [1.0, float(np.sqrt(0.5)), 1.0]
EDGE_WEIGHTS = [1.0, 2.0, 0.5]
# a weight per point of pipe(3, 6), [u][v]
BULGE_WEIGHTS = 1.0 + 0.5 * np.sin(np.arange(18.0)).reshape(3, 6) ** 2
# a periodic cubic's knots over 6 points, uneven, the spacing repeating every 6
LOOP_KNOTS = [-2.5, -1.0, 0.0, 1.0, 3.0, 4.0, 4.5, 6.0, 7.0, 8.0, 10.0]
# Maya's MFnNurbsCurve / MFnNurbsSurface form of a periodic shape
PERIODIC = 3


def wave(count: int, lift: float = 0.0) -> np.ndarray:
    """``count`` points along x, off the axis in y and z"""
    at = np.arange(count, dtype=float)
    return np.stack([at, np.sin(at * 1.3) + lift, np.cos(at * 0.7)], axis=1)


def ring(count: int = 6) -> np.ndarray:
    """``count`` points of one turn about y, climbing"""
    turn = np.arange(count) * (2.0 * np.pi / count)
    return np.stack([2.0 * np.cos(turn), 0.25 * np.arange(count), 2.0 * np.sin(turn)], axis=1)


def sheet(count_u: int, count_v: int) -> np.ndarray:
    """a (count_u, count_v, 3) grid over x and z, uneven in y"""
    u, v = np.meshgrid(np.arange(count_u, dtype=float), np.arange(count_v, dtype=float), indexing="ij")
    return np.stack([u, 0.1 * u * v + np.sin(u + 2.0 * v), v], axis=2)


def pipe(count_u: int, count_v: int) -> np.ndarray:
    """``count_u`` rings of ``count_v`` points up y, widening"""
    u, v = np.meshgrid(np.arange(count_u, dtype=float), np.arange(count_v) * (2.0 * np.pi / count_v), indexing="ij")
    wide = 1.0 + 0.1 * u
    return np.stack([wide * np.cos(v), u, wide * np.sin(v)], axis=2)


def rig() -> HierarchyData:
    """a transform per name of NODES, "knots3" off the origin"""
    made                                  = [TransformData(name) for name in NODES]
    made[NODES.index("knots3")].translate = [1.0, 2.0, 3.0]
    return HierarchyData(made)


def curves(nodes) -> NurbsCurveList:
    """a curve of every kind a file holds, each under the node named for
    it: cubic, open and periodic, on cgmath's knots and on uneven ones;
    degree 2; degree 1 on uneven knots and periodic, which are NURBS curves
    in the file; degree 1 on no knots, a line of an FbxLine, alone and as
    the two of "lines"; a quarter circle and a curve of degree 1 holding
    weights, NURBS curves in the file; and one under no node"""

    def on(name: str) -> str:
        return nodes[name].uuid

    return NurbsCurveList([
        NurbsCurveData(points=wave(6), degree=3, parent_uuid=on("open3")),
        NurbsCurveData(points=wave(6, 1.0), degree=3, knots=[0, 0, 0, 1, 2.5, 3, 3, 3], parent_uuid=on("knots3")),
        NurbsCurveData(points=ring(), degree=3, periodic=True, parent_uuid=on("loop3")),
        NurbsCurveData(points=ring(), degree=3, periodic=True, knots=LOOP_KNOTS, parent_uuid=on("loopKnots")),
        NurbsCurveData(points=wave(5), degree=2, parent_uuid=on("degree2")),
        NurbsCurveData(points=wave(4), degree=1, knots=[0, 1, 3, 4], parent_uuid=on("degree1")),
        NurbsCurveData(points=ring(), degree=1, periodic=True, parent_uuid=on("loop1")),
        NurbsCurveData(points=wave(4, 2.0), degree=1, parent_uuid=on("line")),
        NurbsCurveData(points=wave(2), degree=1, parent_uuid=on("lines")),
        NurbsCurveData(points=wave(3, 1.0), degree=1, parent_uuid=on("lines")),
        NurbsCurveData(points=ARC, degree=2, weights=ARC_WEIGHTS, parent_uuid=on("arc")),
        NurbsCurveData(points=wave(3, 4.0), degree=1, weights=EDGE_WEIGHTS, parent_uuid=on("edge")),
        NurbsCurveData(points=wave(4, 3.0), degree=3),
    ])


def surfaces(nodes) -> NurbsSurfaceList:
    """a bicubic patch, a pipe of degree 1 by 3 periodic in v, a patch of
    degree 2 by 3 on uneven knots, a pipe holding weights, each under the
    node named for it, and a patch under no node"""

    def on(name: str) -> str:
        return nodes[name].uuid

    return NurbsSurfaceList([
        NurbsSurfaceData(
            points=sheet(4, 5), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False, parent_uuid=on("patch")
        ),
        NurbsSurfaceData(
            points=pipe(3, 6), degree_u=1, degree_v=3, periodic_u=False, periodic_v=True, parent_uuid=on("tube")
        ),
        NurbsSurfaceData(
            points=sheet(4, 5) + 1.0, degree_u=2, degree_v=3, periodic_u=False, periodic_v=False,
            knots_u=[0, 0, 0.7, 2, 2], knots_v=[0, 0, 0, 1.5, 2, 2, 2], parent_uuid=on("patchKnots"),
        ),
        NurbsSurfaceData(
            points=pipe(3, 6) + [0.0, 4.0, 0.0], degree_u=1, degree_v=3, periodic_u=False, periodic_v=True,
            weights=BULGE_WEIGHTS, parent_uuid=on("bulge"),
        ),
        NurbsSurfaceData(points=sheet(4, 4), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False),
    ])


def by_node(nodes, shapes, loose: str) -> dict:
    """the shapes under each node's name, in their order; ``loose`` names
    the ones under no node of ``nodes``"""
    found = {}
    for shape in shapes or []:
        name = loose if shape.parent_uuid is None else nodes[shape.parent_uuid].name
        found.setdefault(name, []).append(shape)
    return found


def rig_of(back):
    """the nodes of a read; a file holding a take, as Maya's does, returns
    them as its clip"""
    return back.hierarchy if back.hierarchy is not None else back.clip.active


def same_knots(test, got, want, label: str) -> None:
    """a field that is None or an array, knots or weights: the same, bit
    for bit"""
    test.assertEqual(got is None, want is None, label)
    if want is not None:
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want, dtype=float), err_msg=label)


class _Shapes(_Files):
    def assertSameCurves(self, got, want) -> None:
        """each curve as it was, bit for bit"""
        self.assertEqual(len(got), len(want))
        for at, (a, b) in enumerate(zip(got, want)):
            label = f"curve {at}"
            self.assertEqual(a, b, label)
            np.testing.assert_array_equal(np.asarray(a.points), np.asarray(b.points), err_msg=label)
            self.assertEqual((a.degree, a.periodic), (b.degree, b.periodic), label)
            same_knots(self, a.knots, b.knots, label)
            same_knots(self, a.weights, b.weights, label)

    def assertSameSurfaces(self, got, want) -> None:
        self.assertEqual(len(got), len(want))
        for at, (a, b) in enumerate(zip(got, want)):
            label = f"surface {at}"
            self.assertEqual(a, b, label)
            np.testing.assert_array_equal(np.asarray(a.points), np.asarray(b.points), err_msg=label)
            self.assertEqual((a.degree_u, a.degree_v), (b.degree_u, b.degree_v), label)
            self.assertEqual((a.periodic_u, a.periodic_v), (b.periodic_u, b.periodic_v), label)
            same_knots(self, a.knots_u, b.knots_u, label)
            same_knots(self, a.knots_v, b.knots_v, label)
            same_knots(self, a.weights, b.weights, label)


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestRoundTrip(_Shapes):
    """a save read back is the data saved"""

    def saved(self, **more):
        nodes   = rig()
        given   = curves(nodes)
        patches = surfaces(nodes)
        path    = self.path("shapes.fbx")
        save_fbx(path, nodes, curves=given, surfaces=patches, **more)
        return path, nodes, given, patches

    def test_round_trip(self):
        path, nodes, given, patches = self.saved()
        back = load_fbx(path)
        # a shape under no node returns on a new root node, at identity
        self.assertEqual(list(back.hierarchy.name), list(NODES) + ["curve", "surface"])
        np.testing.assert_array_equal(np.asarray(back.hierarchy["knots3"].translate), [1.0, 2.0, 3.0])
        for name in ("curve", "surface"):
            np.testing.assert_array_equal(np.asarray(back.hierarchy[name].world_matrix), np.eye(4))
        self.assertSameCurves(back.curves, given)
        self.assertSameSurfaces(back.surfaces, patches)
        self.assertEqual(
            {name: len(held) for name, held in by_node(back.hierarchy, back.curves, "").items()},
            {name: len(held) for name, held in by_node(nodes, given, "curve").items()},
        )
        self.assertEqual(
            list(by_node(back.hierarchy, back.surfaces, "")), list(by_node(nodes, patches, "surface"))
        )
        # and what was read saves as it stands
        again = self.path("again.fbx")
        save_fbx(again, *back)
        twice = load_fbx(again)
        self.assertEqual(list(twice.hierarchy.name), list(back.hierarchy.name))
        self.assertSameCurves(twice.curves, back.curves)
        self.assertSameSurfaces(twice.surfaces, back.surfaces)

    def test_ascii(self):
        """an ASCII file holds the sdk's digits: equal, not bit for bit"""
        path, _, given, patches = self.saved(as_ascii=True)
        back = load_fbx(path)
        self.assertEqual(list(back.curves), list(given))
        self.assertEqual(list(back.surfaces), list(patches))

    def test_one_of_each(self):
        """a single curve and a single surface, with no hierarchy"""
        path  = self.path("one.fbx")
        curve = NurbsCurveData(points=wave(6), degree=3)
        patch = NurbsSurfaceData(points=sheet(4, 4), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False)
        save_fbx(path, curves=curve, surfaces=patch)
        back = load_fbx(path)
        self.assertEqual(list(back.hierarchy.name), ["curve", "surface"])
        self.assertSameCurves(back.curves, [curve])
        self.assertSameSurfaces(back.surfaces, [patch])

    def test_lists(self):
        """each list through its own save_fbx and load_fbx"""
        nodes   = rig()
        given   = curves(nodes)
        patches = surfaces(nodes)
        path    = self.path("curves.fbx")
        given.save_fbx(path)
        back = NurbsCurveList.load_fbx(path)
        self.assertIs(type(back), NurbsCurveList)
        self.assertSameCurves(back, given)
        # no hierarchy went with them: a root node each, in the list's order
        self.assertEqual(list(load_fbx(path).hierarchy.name), ["curve"] * len(given))
        none = NurbsSurfaceList.load_fbx(path)
        self.assertIs(type(none), NurbsSurfaceList)
        self.assertEqual(len(none), 0)
        path = self.path("surfaces.fbx")
        patches.save_fbx(path)
        back = NurbsSurfaceList.load_fbx(path)
        self.assertIs(type(back), NurbsSurfaceList)
        self.assertSameSurfaces(back, patches)
        self.assertEqual(list(load_fbx(path).hierarchy.name), ["surface"] * len(patches))
        self.assertEqual(len(NurbsCurveList.load_fbx(path)), 0)
        # ASCII holds the sdk's digits: equal, not bit for bit
        path = self.path("ascii.fbx")
        patches.save_fbx(path, as_ascii=True)
        self.assertEqual(list(NurbsSurfaceList.load_fbx(path)), list(patches))
        # a file of meshes alone holds neither
        path = self.path("mesh.fbx")
        MeshList([plane(None)]).save_fbx(path)
        for kind in (NurbsCurveList, NurbsSurfaceList):
            self.assertIs(type(kind.load_fbx(path)), kind)
            self.assertEqual(len(kind.load_fbx(path)), 0)

    def test_flags(self):
        path, _, given, patches = self.saved()
        back = load_fbx(path, curves=False)
        self.assertIsNone(back.curves)
        self.assertSameSurfaces(back.surfaces, patches)
        back = load_fbx(path, surfaces=False)
        self.assertIsNone(back.surfaces)
        self.assertSameCurves(back.curves, given)
        # the nodes stay, as transforms
        self.assertEqual(list(back.hierarchy.name), list(NODES) + ["curve", "surface"])

    def test_file(self):
        """what the file holds, as Maya's exporter writes it"""
        path, nodes, given, patches = self.saved()
        kinds = FBX.FbxNodeAttribute.EType
        with open_fbx(path, FBX) as scene:
            root = scene.GetRootNode()
            held = {str(root.GetChild(i).GetName()): root.GetChild(i) for i in range(root.GetChildCount())}
            kind = {name: node.GetNodeAttribute().GetAttributeType() for name, node in held.items()}
            for name in ("open3", "knots3", "loop3", "loopKnots", "degree2", "degree1", "loop1", "arc", "edge", "curve"):
                self.assertEqual(kind[name], kinds.eNurbsCurve, name)
            for name in ("line", "lines"):
                self.assertEqual(kind[name], kinds.eLine, name)
            for name in ("patch", "tube", "patchKnots", "bulge", "surface"):
                self.assertEqual(kind[name], kinds.eNurbsSurface, name)
            for name, node in held.items():
                self.assertEqual(node.GetNodeAttributeCount(), 1, name)
                self.assertEqual(str(node.GetNodeAttribute().GetName()), "", name)

            def curve(name: str):
                return typed(held[name].GetNodeAttribute(), FBX.FbxNurbsCurve, FBX)

            forms = FBX.FbxNurbsCurve.EType
            # the knots: ours, which are Maya's, and one more at each end
            self.assertEqual(curve("open3").GetOrder(), 4)
            self.assertEqual(curve("open3").GetType(), forms.eOpen)
            self.assertEqual(list(curve("open3").GetKnotVector()), [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0])
            self.assertEqual(list(curve("knots3").GetKnotVector()), [0.0, 0.0, 0.0, 0.0, 1.0, 2.5, 3.0, 3.0, 3.0, 3.0])
            # a periodic curve: its points once each
            self.assertEqual(curve("loop3").GetType(), forms.ePeriodic)
            self.assertEqual(curve("loop3").GetControlPointsCount(), 6)
            self.assertEqual(list(curve("loop3").GetKnotVector()), [float(k) for k in range(-3, 10)])
            self.assertEqual(list(curve("loopKnots").GetKnotVector())[1:-1], LOOP_KNOTS)
            self.assertEqual(curve("degree2").GetOrder(), 3)
            self.assertEqual(curve("degree1").GetOrder(), 2)
            self.assertEqual(list(curve("degree1").GetKnotVector()), [-1.0, 0.0, 1.0, 3.0, 4.0, 5.0])
            self.assertEqual((curve("loop1").GetOrder(), curve("loop1").GetType()), (2, forms.ePeriodic))

            # a weight: the point's fourth value, the point itself as it stands; 1 where the curve holds none
            got = np.array([tuple(curve("arc").GetControlPointAt(i)) for i in range(3)])
            np.testing.assert_array_equal(got, np.column_stack([ARC, ARC_WEIGHTS]))
            self.assertEqual([tuple(curve("open3").GetControlPointAt(i))[3] for i in range(6)], [1.0] * 6)
            # a curve of degree 1 holding weights is no line, which holds none
            self.assertEqual((curve("edge").GetOrder(), curve("edge").GetControlPointsCount()), (2, 3))
            self.assertEqual([tuple(curve("edge").GetControlPointAt(i))[3] for i in range(3)], EDGE_WEIGHTS)

            # the curves of degree 1 on no knots: the lines of one FbxLine
            line = typed(held["line"].GetNodeAttribute(), FBX.FbxLine, FBX)
            self.assertEqual((line.GetControlPointsCount(), line.GetEndPointCount()), (4, 1))
            lines = typed(held["lines"].GetNodeAttribute(), FBX.FbxLine, FBX)
            self.assertEqual((lines.GetControlPointsCount(), lines.GetEndPointCount()), (5, 2))
            self.assertEqual([lines.GetEndPointAt(i) for i in range(2)], [1, 4])
            got = np.array([tuple(lines.GetControlPointAt(i))[:3] for i in range(5)])
            np.testing.assert_array_equal(got, np.concatenate([wave(2), wave(3, 1.0)]))

            # a surface: its points u fastest
            tube = typed(held["tube"].GetNodeAttribute(), FBX.FbxNurbsSurface, FBX)
            self.assertEqual((tube.GetUCount(), tube.GetVCount()), (3, 6))
            self.assertEqual((tube.GetUOrder(), tube.GetVOrder()), (2, 4))
            sides = FBX.FbxNurbsSurface.EType
            self.assertEqual((tube.GetNurbsUType(), tube.GetNurbsVType()), (sides.eOpen, sides.ePeriodic))
            got = np.array([tuple(tube.GetControlPointAt(i))[:3] for i in range(18)])
            np.testing.assert_array_equal(got, pipe(3, 6).transpose(1, 0, 2).reshape(-1, 3))
            self.assertEqual([tuple(tube.GetControlPointAt(i))[3] for i in range(18)], [1.0] * 18)
            bulge = typed(held["bulge"].GetNodeAttribute(), FBX.FbxNurbsSurface, FBX)
            got   = np.array([tuple(bulge.GetControlPointAt(i))[3] for i in range(18)])
            np.testing.assert_array_equal(got, BULGE_WEIGHTS.T.reshape(-1))
            knots = typed(held["patchKnots"].GetNodeAttribute(), FBX.FbxNurbsSurface, FBX)
            self.assertEqual(list(knots.GetUKnotVector()), [0.0, 0.0, 0.0, 0.7, 2.0, 2.0, 2.0])
            self.assertEqual(list(knots.GetVKnotVector())[1:-1], [0.0, 0.0, 0.0, 1.5, 2.0, 2.0, 2.0])


class TestRefused(_Files):
    """what a .fbx, or Maya's reading of one, has no place for"""

    def refused(self, message: str, **data) -> None:
        with self.assertRaises(ValueError) as caught:
            save_fbx(self.path("refused.fbx"), **data)
        self.assertEqual(str(caught.exception), message)

    def test_two_curves_on_a_node(self):
        nodes = HierarchyData([TransformData("a")])
        cubic = NurbsCurveData(points=wave(6), degree=3, parent_uuid=nodes["a"].uuid)
        line  = NurbsCurveData(points=wave(3), degree=1, parent_uuid=nodes["a"].uuid)
        told = (
            "save_fbx: curve 0 of curves= and curve 1 of curves= sit on node 'a'; a node holds one shape, or the "
            "curves of degree 1 holding no knots as the lines of one"
        )
        # Maya writes such a transform as an empty node, and reads the first curve only of a node of two
        self.refused(told, hierarchy=nodes, curves=NurbsCurveList([cubic, cubic.copy()]))
        self.refused(told, hierarchy=nodes, curves=NurbsCurveList([cubic, line]))
        self.refused(told, hierarchy=nodes, curves=NurbsCurveList([line, cubic]))

    def test_beside_another_shape(self):
        nodes = HierarchyData([TransformData("a")])
        curve = NurbsCurveData(points=wave(6), degree=3, parent_uuid=nodes["a"].uuid)
        patch = NurbsSurfaceData(
            points=sheet(4, 4), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False, parent_uuid=nodes["a"].uuid
        )
        self.refused(
            "save_fbx: curve 0 of curves= and surface 0 of surfaces= sit on node 'a'; a node holds one shape, or "
            "the curves of degree 1 holding no knots as the lines of one",
            hierarchy=nodes, curves=curve, surfaces=patch,
        )
        with self.assertRaisesRegex(ValueError, "and curve 0 of curves= sit on node 'a'; a node holds one shape"):
            save_fbx(self.path("refused.fbx"), nodes, meshes=MeshList([plane(nodes["a"].uuid)]), curves=curve)

    def test_points_not_3d(self):
        self.refused(
            "save_fbx: curve 0 of curves= holds points of 2 components; a .fbx holds 3D points",
            curves=NurbsCurveData(points=wave(6)[:, :2], degree=3),
        )

    def test_weights(self):
        """weights that are not one a control point, finite and above zero"""
        self.refused(
            "save_fbx: curve 0 of curves=: weights must hold one value per control point, shape (3,), got shape (2,)",
            curves=NurbsCurveData(points=ARC, degree=2, weights=[1.0, 2.0]),
        )
        patch = NurbsSurfaceData(
            points=sheet(4, 4), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False, weights=np.zeros((4, 4))
        )
        self.refused("save_fbx: surface 0 of surfaces=: weights must be finite and above zero", surfaces=patch)

    def test_on_a_joint(self):
        nodes = HierarchyData([TransformData("j", node_type="joint")])
        self.refused(
            "save_fbx: curve 0 of curves= sits on joint 'j'; Maya's FBX holds a curve on a transform only",
            hierarchy=nodes, curves=NurbsCurveData(points=wave(6), degree=3, parent_uuid=nodes["j"].uuid),
        )
        patch = NurbsSurfaceData(
            points=sheet(4, 4), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False, parent_uuid=nodes["j"].uuid
        )
        self.refused(
            "save_fbx: surface 0 of surfaces= sits on joint 'j'; Maya's FBX holds a surface on a transform only",
            hierarchy=nodes, surfaces=patch,
        )

    def test_wrong_class(self):
        nodes = rig()
        with self.assertRaises(TypeError) as caught:
            save_fbx(self.path("refused.fbx"), nodes, curves=surfaces(nodes))
        self.assertEqual(
            str(caught.exception), "save_fbx: the curves are a NurbsCurveList or a NurbsCurveData, not a NurbsSurfaceList"
        )
        with self.assertRaises(TypeError) as caught:
            save_fbx(self.path("refused.fbx"), nodes, surfaces=curves(nodes))
        self.assertEqual(
            str(caught.exception),
            "save_fbx: the surfaces are a NurbsSurfaceList or a NurbsSurfaceData, not a NurbsCurveList",
        )


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMaya(_Recorded, _Shapes):
    """Maya 2025's FBXExport of a scene of curves and surfaces, and its
    FBXImport of a save, as recorded"""

    def same_points(self, got, maya, label: str) -> None:
        np.testing.assert_allclose(np.asarray(got), maya, rtol=0, atol=1e-12, err_msg=label)

    def test_maya_export(self):
        """Maya's file reads as the scene Maya held"""
        held  = self.case("nurbs_export")
        back  = load_fbx(self.fixture("maya.fbx"))
        rig   = rig_of(back)
        found = by_node(rig, back.curves, "")
        for name, shapes in by_node(rig, back.surfaces, "").items():
            found.setdefault(name, []).extend(shapes)
        names = held.value("nodes")
        self.assertEqual(sorted(found), sorted(names))
        for name in names:
            maya = held.value(name)
            self.same_points(rig[name].translate, maya["translate"], name)
            self.assertEqual(len(found[name]), len(maya["shapes"]), name)
            for at, (shape, told) in enumerate(zip(found[name], maya["shapes"])):
                key = f"{name}/{at}"
                cvs = held.array(f"{key}/cvs")
                # Maya's weights, each 1, read as none
                np.testing.assert_array_equal(held.array(f"{key}/weights"), 1.0, err_msg=key)
                self.assertIsNone(shape.weights, key)
                if told["type"] == "nurbsSurface":
                    self.assertEqual((shape.degree_u, shape.degree_v), (told["degree_u"], told["degree_v"]), key)
                    self.assertEqual(shape.periodic_u, told["form_u"] == PERIODIC, key)
                    self.assertEqual(shape.periodic_v, told["form_v"] == PERIODIC, key)
                    # Maya's points: u by v, a periodic direction's first ones repeated at its end
                    grid = cvs.reshape(told["count_u"], told["count_v"], 3)
                    count_u, count_v = np.asarray(shape.points).shape[:2]
                    self.same_points(shape.points, grid[:count_u, :count_v],     key)
                    self.same_points(shape.kv_u,   held.array(f"{key}/knots_u"), key)
                    self.same_points(shape.kv_v,   held.array(f"{key}/knots_v"), key)
                    continue
                self.assertEqual(shape.degree, told["degree"], key)
                if shape.degree == 1:
                    # an FbxLine: no knots, no form; a periodic one's first point returns at its end
                    self.assertFalse(shape.periodic, key)
                    self.assertIsNone(shape.knots, key)
                    self.same_points(shape.points, cvs, key)
                    continue
                self.assertEqual(shape.periodic, told["form"] == PERIODIC, key)
                self.same_points(shape.points, cvs[: len(shape.points)], key)
                self.same_points(shape.kv, held.array(f"{key}/knots"), key)
        # the cases the scene was built for
        self.assertEqual(held.value("loop3")["shapes"][0]["form"], PERIODIC)
        self.assertEqual(held.value("closed3")["shapes"][0]["form"], 2)
        np.testing.assert_array_equal(held.array("closed3/0/cvs")[0], held.array("closed3/0/cvs")[-1])
        self.assertEqual(len(found["closed3"][0].points), len(held.array("closed3/0/cvs")))
        same_knots(self, found["loopKnots"][0].knots, LOOP_KNOTS, "loopKnots")
        self.assertIsNone(found["loop3"][0].knots)
        self.assertEqual(held.value("loop1")["shapes"][0]["form"], PERIODIC)
        # a curve of degree 1 loses its knots in Maya's file
        np.testing.assert_array_equal(held.array("lineKnots/0/knots"), [0.0, 1.0, 3.0, 4.0])
        np.testing.assert_array_equal(found["lineKnots"][0].kv, [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(len(found["lines"]), 2)
        self.assertTrue(PERIODIC in (held.value("tube")["shapes"][0]["form_u"], held.value("tube")["shapes"][0]["form_v"]))

    def test_maya_import(self):
        """a save, as Maya read it"""
        held    = self.case("nurbs_import")
        nodes   = rig()
        given   = curves(nodes)
        patches = surfaces(nodes)
        self.assertFreshSave(self.fixture("ours.fbx"), hierarchy=nodes, curves=given, surfaces=patches)
        self.same_points([1.0, 2.0, 3.0], held.value("knots3")["translate"], "knots3")
        for name, shapes in by_node(nodes, given, "curve").items():
            maya = held.value(name)["shapes"]
            self.assertEqual([told["type"] for told in maya], ["nurbsCurve"] * len(shapes), name)
            for at, (shape, told) in enumerate(zip(shapes, maya)):
                key = f"{name}/{at}"
                cvs = held.array(f"{key}/cvs")
                self.assertEqual(told["degree"], shape.degree, key)
                self.assertEqual(told["form"] == PERIODIC, shape.periodic, key)
                self.same_points(shape.points, cvs[: len(shape.points)], key)
                self.same_points(shape.kv, held.array(f"{key}/knots"), key)
                count   = len(shape.points)
                weights = np.ones(count) if shape.weights is None else shape.weights
                self.same_points(weights, held.array(f"{key}/weights")[:count], key)
                if shape.weights is not None:
                    # the curve Maya made of them: the points Maya finds on it
                    got = shape.compute(held.array(f"{key}/at"))[0]
                    np.testing.assert_allclose(got, held.array(f"{key}/samples"), rtol=0, atol=1e-6, err_msg=key)
                if shape.periodic:
                    # Maya's own count: the first points again at the end
                    self.same_points(np.asarray(shape.points)[: shape.degree], cvs[len(shape.points) :], key)
        for name, shapes in by_node(nodes, patches, "surface").items():
            maya = held.value(name)["shapes"]
            self.assertEqual([told["type"] for told in maya], ["nurbsSurface"], name)
            shape, told, key = shapes[0], maya[0], f"{name}/0"
            self.assertEqual((told["degree_u"], told["degree_v"]), (shape.degree_u, shape.degree_v), key)
            self.assertEqual((told["form_u"] == PERIODIC, told["form_v"] == PERIODIC), (shape.periodic_u, shape.periodic_v), key)
            grid = held.array(f"{key}/cvs").reshape(told["count_u"], told["count_v"], 3)
            count_u, count_v = np.asarray(shape.points).shape[:2]
            self.same_points(shape.points, grid[:count_u, :count_v],     key)
            self.same_points(shape.kv_u,   held.array(f"{key}/knots_u"), key)
            self.same_points(shape.kv_v,   held.array(f"{key}/knots_v"), key)
            weights      = np.ones((count_u, count_v)) if shape.weights is None else shape.weights
            maya_weights = held.array(f"{key}/weights").reshape(told["count_u"], told["count_v"])
            self.same_points(weights, maya_weights[:count_u, :count_v], key)
            if shape.weights is not None:
                at  = held.array(f"{key}/at")
                got = shape.evaluate(at[:, 0], at[:, 1])
                np.testing.assert_allclose(got, held.array(f"{key}/samples"), rtol=0, atol=1e-6, err_msg=key)

    def test_rational(self):
        """Maya's rational curve and surface: the weights Maya held, the
        points as Maya holds them, and the shape Maya's own"""
        held = self.case("nurbs_export")
        back = load_fbx(self.fixture("maya_rational.fbx"))
        rig  = rig_of(back)
        self.assertEqual(sorted(rig.name), ["patch", "rational"])
        curve = by_node(rig, back.curves, "")["rational"][0]
        np.testing.assert_array_equal(curve.weights, [1.0, 0.5, 1.0])
        np.testing.assert_array_equal(curve.weights, held.array("rational/0/weights"))
        # a point is the file's as it stands, which is Maya's cv: not multiplied by its weight
        np.testing.assert_array_equal(np.asarray(curve.points)[1], [1.0, 1.0, 0.0])
        self.same_points(curve.points, held.array("rational/0/cvs"), "rational")
        got = curve.compute(held.array("rational/0/at"))[0]
        np.testing.assert_allclose(got, held.array("rational/0/samples"), rtol=0, atol=1e-6)
        # Maya's pointOnCurve finds the same point
        np.testing.assert_allclose(curve.compute(0.5)[0], held.array("rational/command"), rtol=0, atol=1e-6)
        # the plain surface beside it holds no weights
        self.assertIsNone(back.surfaces[0].weights)

        back = load_fbx(self.fixture("maya_rational_surface.fbx"))
        rig  = rig_of(back)
        self.assertEqual(sorted(rig.name), ["plain", "rationalPatch"])
        patch      = by_node(rig, back.surfaces, "")["rationalPatch"][0]
        want       = np.ones((4, 4))
        want[1, 1] = 2.0
        np.testing.assert_array_equal(patch.weights, want)
        np.testing.assert_array_equal(patch.weights.reshape(-1), held.array("rationalPatch/0/weights"))
        self.same_points(np.asarray(patch.points).reshape(-1, 3), held.array("rationalPatch/0/cvs"), "rationalPatch")
        at  = held.array("rationalPatch/0/at")
        got = patch.evaluate(at[:, 0], at[:, 1])
        np.testing.assert_allclose(got, held.array("rationalPatch/0/samples"), rtol=0, atol=1e-6)
        np.testing.assert_allclose(patch.evaluate(0.5, 0.5), held.array("rationalPatch/command"), rtol=0, atol=1e-6)
        self.assertIsNone(back.curves[0].weights)

    def test_trimmed(self):
        self.case("nurbs_export")
        path = self.fixture("maya_trimmed.fbx")
        with self.assertRaisesRegex(
            ValueError,
            r"^load_fbx: the surface on node 'trimmed' is trimmed, which the r17-skeleton fork does not read; "
            r"load_fbx\(\.\.\., surfaces=False\) reads the file without its surfaces$",
        ):
            load_fbx(path)
        back = load_fbx(path, surfaces=False)
        self.assertEqual(sorted(rig_of(back).name), ["plain", "trimmed"])
        self.assertEqual(len(back.curves), 1)


if __name__ == "__main__":
    unittest.main()
