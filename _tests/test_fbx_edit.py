"""FbxFile.save of changed records: each change written into the open file.

A record changed on an open handle is written into the file's own scene and
saved with it: the saved file reads as the handle's records hold, in the
file's own axes, unit and pivots (Maya's Z-up files in cm and in metres are
among the files). A change the file cannot hold is refused, naming what to
do, nothing written and the open scene put back as the file; with
force=True a change may drop data no record holds, a warning naming each
item. The tests need the Autodesk SDK; the Maya files were written by Maya
2025, recorded once (maya_fixtures), and need no Maya.
"""

import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
from cgmath._tests.maya_fixtures import same_read
from cgmath._tests.test_fbx_file import FBX, FIELDS, MAYA_FILES, _Handles, census, every_kind, maya_file
from cgmath.formats import FbxContents, FbxFile, load_fbx, save_fbx

MOVES  = ("translate", "rotate", "scale", "joint_orient", "rotate_axis")
LABELS = ("name", "rotate_order", "visibility", "segment_scale_compensate", "radius", "draw_style", "node_type")

# how near a saved value reads back: a take's keys go through the 32-bit
# floats an fbx curve holds
RTOL   = 1e-6
ATOL   = 1e-4

SHARED = {}


def setUpModule():
    if FBX is not None:
        SHARED["folder"] = tempfile.mkdtemp(prefix="cgmath_fbx_edit_")
        SHARED["every"]  = every_kind(os.path.join(SHARED["folder"], "every.fbx"))


def tearDownModule():
    shutil.rmtree(SHARED.pop("folder", ""), True)


def loaded(path: str) -> FbxContents:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_fbx(path, verbose=False)


def everything(f: FbxFile) -> FbxContents:
    return FbxContents(*(getattr(f, field) for field in FIELDS))


def named(f: FbxFile, name: str):
    return next(mesh for mesh in f.meshes if mesh.name == name)


def lifted(shape, rows=slice(None), by=(0.25, 0.5, -0.125)) -> None:
    """``shape``'s points, or some rows of them, moved"""
    points = np.array(shape.points)
    points[rows] += by
    shape.points = points


def keyed(take, channel: str, node: int, by) -> None:
    """every frame of one node of ``take`` moved"""
    values = np.array(getattr(take.frames, channel))
    values[:, node] += by
    setattr(take.frames, channel, values)


def moves(what: str):
    """a change of the first and the last node of a file: ``t`` their
    translates, ``r`` their rotates, ``n`` the last one's name, ``v`` its
    visibility over the second half of each take"""

    def edit(f):
        clip = f.clip
        if clip is None:
            nodes = f.hierarchy.list
            for node in (nodes[0], nodes[-1]):
                if "t" in what:
                    node.translate = np.asarray(node.translate) + [1.0, -2.0, 0.5]
                if "r" in what:
                    node.rotate = np.asarray(node.rotate) + [10.0, 5.0, -2.5]
            return
        for take in clip:
            count = take.frame_count
            ramp  = np.linspace(0.0, 1.0, count)[:, None]
            last  = len(take.list) - 1
            if "t" in what:
                keyed(take, "translate", 0, ramp * [1.0, -2.0, 0.5])
                keyed(take, "translate", last, [0.25, 0.5, -0.75])
            if "r" in what:
                keyed(take, "rotate", 0, ramp * [10.0, 5.0, -2.5])
                keyed(take, "rotate", last, [3.0, -4.0, 5.0])
            if "n" in what:
                take.list[last].name = take.list[last].name + "_x"
            if "v" in what:
                shown                     = np.array(take.frames.visibility)
                shown[count // 2 :, last] = ~shown[count // 2 :, last]
                take.frames.visibility    = shown

    return edit


class _Edits(_Handles):
    def assertNodes(self, got, want, fields, where: str) -> None:
        self.assertEqual([node.name for node in got], [node.name for node in want], where)
        self.assertEqual([node.parent_name for node in got], [node.parent_name for node in want], where)
        for one, other in zip(got.list, want.list):
            for field in fields:
                a, b = getattr(one, field), getattr(other, field)
                if isinstance(a, np.ndarray):
                    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, err_msg=f"{where} {one.name}.{field}")
                else:
                    self.assertEqual(a, b, f"{where} {one.name}.{field}")

    def assertTake(self, got, want, where: str) -> None:
        spans = [(take.frame_count, take.start_frame, take.fps) for take in (got, want)]
        self.assertEqual(spans[0], spans[1], where)
        for channel in ("translate", "rotate", "scale"):
            np.testing.assert_allclose(
                np.asarray(getattr(got.frames, channel)),
                np.asarray(getattr(want.frames, channel)),
                rtol    = RTOL,
                atol    = ATOL,
                err_msg = f"{where}.{channel}",
            )
        np.testing.assert_array_equal(np.asarray(got.frames.visibility), np.asarray(want.frames.visibility), where)
        np.testing.assert_array_equal(np.asarray(got.keyed), np.asarray(want.keyed), where)
        self.assertNodes(got, want, LABELS + ("joint_orient", "rotate_axis"), where)

    def assertNear(self, got, want, label: str = "") -> None:
        """two reads hold the same records, uuids aside: a node's values
        and a take's keys within the floats a file holds them in, every
        shape equal and of the same name"""
        for field in FIELDS:
            a, b = getattr(got, field), getattr(want, field)
            where = f"{label}{field}"
            if a is None or b is None:
                self.assertIs(a, b, where)
                continue
            if field == "clip":
                self.assertEqual(list(a.take), list(b.take), where)
                self.assertEqual([take is a.active for take in a], [take is b.active for take in b], where)
                for x, y in zip(a, b):
                    self.assertTake(x, y, where)
                continue
            if field == "hierarchy":
                self.assertNodes(a, b, LABELS + MOVES, where)
                for one, other in zip(a.list, b.list):
                    self.assertEqual(one.user_defined_attributes, other.user_defined_attributes, f"{where} {one.name}")
                continue
            self.assertEqual(len(a), len(b), where)
            for i, (x, y) in enumerate(zip(a, b)):
                self.assertEqual(getattr(x, "name", None), getattr(y, "name", None), f"{where}[{i}]")
                if field in ("curves", "surfaces"):
                    # a shape given other points keeps the tables it built of the old ones; a copy builds its own
                    x, y = x.copy(), y.copy()
                self.assertEqual(x, y, f"{where}[{i}]")

    def assertSaves(self, source: str, edit, force: bool = False, name: str = "saved.fbx") -> list:
        """``source`` opened, ``edit(f)`` applied, the handle saved: the
        saved file reads as the handle's records. Returns the save's
        warnings"""
        out = self.path(name)
        with FbxFile(source, verbose=False) as f:
            edit(f)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                f.save(out, force=force)
            self.assertNear(loaded(out), everything(f), "saved: ")
        return [str(warning.message) for warning in caught]

    def assertRefuses(self, source: str, edit, said: str, force: bool = False) -> None:
        """the save of ``edit`` is refused, naming ``said``, and writes
        nothing"""
        before = sorted(os.listdir(self.folder))
        with FbxFile(source, verbose=False) as f:
            edit(f)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaises(ValueError) as caught:
                    f.save(self.path("refused.fbx"), force=force)
        self.assertIn(said, str(caught.exception))
        self.assertTrue(str(caught.exception).startswith("FbxFile.save: "), str(caught.exception))
        self.assertEqual(sorted(os.listdir(self.folder)), before)

    def assertEach(self, saves=(), refusals=()) -> None:
        """each ``(source, edit)`` or ``(source, edit, force)`` saves, a
        forced one warning of what it drops and no other; each ``(source,
        edit, said)`` or ``(source, edit, said, force)`` is refused"""
        for source, edit, *force in saves:
            with self.subTest(saves=edit.__name__):
                told   = self.assertSaves(source, edit, force=bool(force), name=f"{edit.__name__}.fbx")
                forced = [text for text in told if text.startswith("FbxFile.save: with force")]
                self.assertEqual(len(forced), 1 if force else 0, told)
        for source, edit, said, *force in refusals:
            with self.subTest(refuses=edit.__name__):
                self.assertRefuses(source, edit, said, force=bool(force))


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestNodes(_Edits):
    """a node's values and a take's keys, in the file's own axes, unit and
    pivots"""

    def test_maya_files(self):
        """the first and the last node of every Maya file moved, turned,
        renamed and hidden; a node holding a skinned mesh stays on its bind
        world"""
        bound = {"uvs_export", "zup_skinned"}
        for case, name in MAYA_FILES:
            for what in ("t", "r", "nv"):
                with self.subTest(case=case, name=name, what=what):
                    source = maya_file(case, name)
                    if case in bound and what != "nv":
                        self.assertRefuses(source, moves(what), "off its bind world")
                    else:
                        self.assertSaves(source, moves(what), name=f"{case}_{what}.fbx")

    def test_pivots(self):
        """every node of a file of pivots turned, its last one scaled"""

        def edit(f):
            take = f.clip[0]
            keyed(take, "rotate", slice(None), [12.0, -7.0, 3.0])
            scale = np.array(take.frames.scale)
            scale[:, -1] *= 2.0
            take.frames.scale = scale

        self.assertSaves(maya_file("pivots_export", "maya.fbx"), edit)

    def test_takes(self):
        source = maya_file("takes_each", "ours.fbx")

        def active(f):
            f.clip.active = f.clip[2]

        def second(f):
            keyed(f.clip[1], "translate", 0, [0.0, 3.0, 0.0])
            f.clip[1].take = "bee"

        def rates(f):
            for take in f.clip:
                take.fps = 30.0

        def one_rate(f):
            f.clip[1].fps = 30.0

        for edit in (active, second, rates):
            with self.subTest(edit=edit.__name__):
                self.assertSaves(source, edit, name=f"{edit.__name__}.fbx")
        self.assertRefuses(source, one_rate, "a file holds one frame rate")
        with FbxFile(self.path("second.fbx"), verbose=False) as f:
            self.assertEqual(list(f.clip.take)[1], "bee")

    def test_retime(self):
        """a take moved in time: the keys of what no record holds (a blend
        shape's weight curve, another writer's keyed camera) would stay
        behind, so the save asks for force, and warns of each with it"""

        def later(f):
            f.clip[0].start_frame = f.clip[0].start_frame + 7

        self.assertSaves(maya_file("views_export", "maya.fbx"), later, name="views.fbx")
        source = maya_file("morphs_export", "maya.fbx")
        self.assertRefuses(source, later, "force=True")
        told = self.assertSaves(source, later, force=True)
        self.assertEqual(len(told), 1)
        self.assertIn("FbxFile.save: with force", told[0])
        other = maya_file("other_writer", "other.fbx")
        self.assertEach(saves=[(other, later, True)], refusals=[(other, later, "force=True")])


def spread(skin) -> None:
    """half of every point's first weight moved to its second influence"""
    weights = np.array(skin.weights)
    moved   = weights[:, 0] * 0.5
    weights[:, 0] -= moved
    weights[:, 1] += moved
    skin.weights = weights


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestShapes(_Edits):
    """the values of a mesh, a UV set, a skin, a blend shape target, a
    camera, a light, a curve and a surface"""

    def test_meshes(self):
        every = SHARED["every"]
        two   = maya_file("two_meshes",    "ours.fbx")
        share = maya_file("colors_export", "shared.fbx")
        morph = maya_file("morphs_export", "maya.fbx")

        def renamed(f):
            named(f, "planeShape").name = "plane_renamed"

        def hard_edged(f):
            lifted(named(f, "tetra"), slice(0, 2))

        def authored(f):
            lifted(named(f, "body"), slice(0, 20))

        def skinned(f):
            lifted(f.meshes[0], slice(10, 30))

        def under_targets(f):
            lifted(named(f, "planeShape"), slice(0, 3))

        def under_maya_targets(f):
            lifted(f.meshes[0], slice(0, 3))

        def soft(f):
            named(f, "tetra").hard_edges = None

        def fewer_hard(f):
            mesh = f.meshes[1]
            self.assertIsNotNone(mesh.hard_edges)
            mesh.hard_edges = np.asarray(mesh.hard_edges)[:4]

        def normals(f):
            mesh                = named(f, "tetra")
            mesh.hard_edges     = None
            mesh.normals        = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
            mesh.normal_indices = np.arange(len(mesh.indices)) % 2

        def no_normals(f):
            mesh                = named(f, "body")
            mesh.normals        = None
            mesh.normal_indices = None

        def colours(f):
            colors             = np.array(f.meshes[0].colors)
            colors[:, 0]       = 1.0 - colors[:, 0]
            f.meshes[0].colors = colors

        def vertex_colours(f):
            mesh               = f.meshes[0]
            mesh.color_indices = None
            mesh.colors        = np.linspace(0.0, 1.0, len(mesh.points) * 4).reshape(-1, 4)

        def no_colours(f):
            f.meshes[0].color_indices = None
            f.meshes[0].colors        = None

        def new_colours(f):
            mesh        = named(f, "tetra")
            mesh.colors = np.linspace(0.0, 1.0, len(mesh.points) * 4).reshape(-1, 4)

        def other_faces(f):
            mesh         = named(f, "tetra")
            mesh.indices = np.asarray(mesh.indices)[::-1].copy()

        self.assertEach(
            saves=[
                (every, renamed),
                (two, hard_edged),
                (two, authored),
                (maya_file("skin_geo", "maya.fbx"), skinned),
                (maya_file("zup_skinned", "z_cm.fbx"), skinned),
                (every, under_targets, True),
                (morph, under_maya_targets, True),
                (two, soft),
                (morph, fewer_hard),
                (two, normals),
                (two, no_normals),
                (share, colours),
                (share, vertex_colours),
                (share, no_colours),
                (two, new_colours),
            ],
            refusals=[
                (every, under_targets, "force=True"),
                (two, other_faces, "other faces"),
            ],
        )

    def test_uvs(self):
        every = SHARED["every"]
        uvs   = maya_file("uvs_export", "maya.fbx")

        def renamed(f):
            f.uvs[1].name = "second_renamed"

        def moved(f):
            f.uvs[0].points = np.asarray(f.uvs[0].points) * 0.5 + 0.25

        def unwelded(f):
            uv         = f.uvs[1]
            corners    = np.asarray(uv.points)[np.asarray(uv.indices)]
            uv.points  = corners
            uv.indices = np.arange(len(corners), dtype=np.int32)

        def flipped(f):
            f.uvs[0].points = 1.0 - np.asarray(f.uvs[0].points)

        def other_faces(f):
            uv         = f.uvs[0]
            uv.indices = np.asarray(uv.indices)[:-4]
            uv.counts  = np.asarray(uv.counts)[:-1]

        self.assertEach(
            saves    = [(uvs, renamed), (uvs, moved), (uvs, unwelded), (every, flipped)],
            refusals = [(uvs, other_faces, "maps every face")],
        )

    def test_skins(self):
        every = SHARED["every"]
        skin  = maya_file("skin_geo", "maya.fbx")

        def weights(f):
            spread(f.skins[0])

        def renamed(f):
            f.skins[0].name = "skin_renamed"

        def one_less(f):
            held    = f.skins[0]
            weights = np.array(held.weights)
            weights[:, 1] += weights[:, 0]
            held.influences = list(held.influences)[1:]
            held.weights    = weights[:, 1:]

        def reordered(f):
            held            = f.skins[0]
            order           = list(range(len(held.influences)))[::-1]
            held.influences = [held.influences[k] for k in order]
            held.weights    = np.asarray(held.weights)[:, order]

        def joint_renamed(f):
            held = f.skins[0]
            name = held.influences[2]
            next(node for node in f.hierarchy.list if node.name == name).name = "renamed_joint"
            held.influences = [("renamed_joint" if x == name else x) for x in held.influences]
            spread(held)

        def joint_renamed_alone(f):
            held = f.skins[0]
            next(node for node in f.hierarchy.list if node.name == held.influences[2]).name = "renamed_joint"
            spread(held)

        def nobody(f):
            f.skins[0].influences = list(f.skins[0].influences)[:-1] + ["nobody"]

        def spare(f):
            held            = f.skins[0]
            other           = next(node.name for node in f.hierarchy.list if node.name not in held.influences)
            held.influences = list(held.influences)[:-1] + [other]

        self.assertEach(
            saves=[
                (skin, weights),
                (every, weights),
                (skin, renamed),
                (skin, one_less),
                (skin, reordered),
                (skin, joint_renamed),
                (skin, joint_renamed_alone),
            ],
            refusals=[(skin, nobody, "names no node"), (skin, spare, "does not hold")],
        )

    def test_morphs(self):
        every = SHARED["every"]
        morph = maya_file("morphs_export", "maya.fbx")

        def weight(f):
            f.morphs[2].weight = 0.4

        def maya_weight(f):
            f.morphs[3].weight = 0.8

        def between_weight(f):
            f.morphs[0].weight = 0.9

        def channel(f):
            f.morphs[0].name = "grin"
            f.morphs[1].name = "grin"

        def half_channel(f):
            f.morphs[0].name = "grin"

        def deformer(f):
            for target in f.morphs[:4]:
                target.deformer = "face"

        def reached(f):
            f.morphs[0].at = 0.25

        def past(f):
            f.morphs[0].at = 1.5

        def offsets(f):
            f.morphs[2].offsets = np.asarray(f.morphs[2].offsets) * 2.0

        def other_points(f):
            f.morphs[3].indices = np.array([0, 5], dtype=np.int32)
            f.morphs[3].offsets = np.array([[0.0, 1.0, 0.0], [0.5, 0.0, -0.25]])

        def points_and_offsets(f):
            mesh = next(mesh for mesh in f.meshes if mesh.uuid == f.morphs[2].mesh_uuid)
            lifted(mesh, slice(0, 3), (0.0, 0.5, 0.0))
            f.morphs[2].offsets = np.asarray(f.morphs[2].offsets) * 0.5

        self.assertEach(
            saves=[
                (every, weight),
                (morph, maya_weight),
                (morph, channel),
                (every, deformer),
                (every, reached),
                (every, offsets, True),
                (morph, other_points, True),
                (every, points_and_offsets, True),
            ],
            refusals=[
                (every, between_weight, "a channel holds one weight"),
                (morph, half_channel, "holds one full target"),
                (every, past, "above its full target"),
                (every, offsets, "force=True"),
            ],
        )

    def test_cameras(self):
        every = SHARED["every"]
        views = maya_file("views_export", "maya.fbx")
        other = maya_file("other_writer", "other.fbx")

        def lens(f):
            camera                          = f.cameras[0]
            camera.focal_length             = 85.0
            camera.horizontal_film_aperture = 1.2
            camera.vertical_film_aperture   = 0.8
            camera.near_clip_plane          = 0.5
            camera.far_clip_plane           = 5000.0
            camera.film_offset              = [0.1, -0.2]
            camera.lens_squeeze_ratio       = 2.0
            camera.film_fit                 = "overscan"

        def renamed(f):
            f.cameras[2].name = "renamedShape"

        def width(f):
            f.cameras[1].orthographic_width = 80.0

        def orthographic(f):
            f.cameras[2].orthographic       = True
            f.cameras[2].orthographic_width = 25.0

        def view_lens(f):
            f.cameras[0].focal_length = 24.0

        def view_back(f):
            f.cameras[0].horizontal_film_aperture = 1.5

        def keyed_lens(f):
            f.cameras[1].focal_length = 24.0

        def wide(f):
            f.cameras[0].focal_length    = 20.0
            f.cameras[0].near_clip_plane = 1.0

        def scaled(f):
            f.cameras[0].camera_scale = 2.0

        self.assertEach(
            saves=[
                (views, lens),
                (views, renamed),
                (views, width),
                (views, orthographic),
                (other, view_lens),
                (other, view_back),
                (every, wide),
            ],
            refusals=[
                (other, keyed_lens, "a take of the file keys"),
                (views, scaled, "camera_scale"),
            ],
        )

    def test_lights(self):
        every = SHARED["every"]
        views = maya_file("views_export", "maya.fbx")
        other = maya_file("other_writer", "other.fbx")

        def of_kind(f, kind):
            return next(light for light in f.lights if light.kind == kind)

        def shade(f):
            for light in f.lights:
                light.color      = [0.25, 0.5, 0.75]
                light.intensity  = 3.5
                light.decay_rate = 1

        def cone(f):
            of_kind(f, "spot").cone_angle = 30.0
            of_kind(f, "spot").penumbra_angle = 12.5

        def renamed(f):
            f.lights[0].name      = "fill"
            f.lights[0].intensity = 0.5

        def other_colour(f):
            f.lights[0].color = [1.0, 0.5, 0.0]

        def kind(f):
            of_kind(f, "spot").kind = "directional"

        def keyed_intensity(f):
            of_kind(f, "point").intensity = 9.0

        self.assertEach(
            saves=[(views, shade), (views, cone), (every, renamed), (other, other_colour)],
            refusals=[
                (views, kind, "keeps its kind"),
                (other, keyed_intensity, "a take of the file keys"),
            ],
        )

    def test_curves_and_surfaces(self):
        """points and weights written into the file's shapes; another
        degree, knot vector or point count asks for force, which builds the
        shape anew, no shape left behind"""
        every = SHARED["every"]
        maya  = maya_file("nurbs_export", "maya.fbx")
        ratio = maya_file("nurbs_export", "maya_rational.fbx")
        rsurf = maya_file("nurbs_export", "maya_rational_surface.fbx")
        by    = (0.25, 0.5, -0.75)

        def a_line(f):
            return next(curve for curve in f.curves if curve.degree == 1 and curve.knots is None)

        def both(f):
            lifted(f.curves[0], by=by)
            lifted(f.surfaces[0], by=by)

        def curves(f):
            for curve in f.curves:
                lifted(curve, by=by)

        def surfaces(f):
            for surface in f.surfaces:
                lifted(surface, by=by)

        def line(f):
            lifted(a_line(f), 1, (0.0, 2.0, 0.0))

        def weights(f):
            f.curves[0].weights = np.asarray(f.curves[0].weights) * 1.5

        def surface_weights(f):
            surface         = f.surfaces[0]
            held            = np.array(surface.weights)
            held[1, 2]      = 3.0
            surface.weights = held
            lifted(surface, by=by)

        def new_weights(f):
            f.curves[0].weights = np.linspace(1.0, 2.0, len(f.curves[0].points))

        def no_weights(f):
            f.curves[0].weights = None

        def knotted(f):
            f.curves[0].knots = np.array([0.0, 0.0, 0.0, 1.0, 2.5, 3.0, 3.0, 3.0])

        def longer(f):
            a_line(f).points = np.concatenate([np.asarray(a_line(f).points), [[9.0, 9.0, 9.0]]])

        def lower(f):
            f.surfaces[0].degree_u = 2

        def other_node(f):
            f.curves[0].parent_uuid = f.surfaces[0].parent_uuid

        rebuilt = [(rsurf, knotted), (maya, longer), (ratio, lower)]
        self.assertEach(
            saves=[
                (every, both),
                (maya, curves),
                (maya, surfaces),
                (maya, line),
                (ratio, weights),
                (rsurf, surface_weights),
                (rsurf, new_weights),
                (ratio, no_weights),
            ]
            + [(source, edit, True) for source, edit in rebuilt],
            refusals=[(source, edit, "force=True") for source, edit in rebuilt]
            + [(ratio, other_node, "a node holds one shape")],
        )
        for source, edit in rebuilt:
            self.assertEqual(census(self.path(f"{edit.__name__}.fbx"))[0], census(source)[0], edit.__name__)


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestMaterials(_Edits):
    """a material's values, its name, its faces, the UV set its maps read"""

    def built(self, name: str, source: str, change) -> str:
        """``source``'s records, changed, written by save_fbx"""
        held = loaded(source)
        change(held)
        path = self.path(name)
        save_fbx(path, *held)
        return path

    def test_values(self):
        every    = SHARED["every"]
        embedded = maya_file("materials_export", "maya_embedded.fbx")
        linked   = maya_file("materials_export", "maya_linked.fbx")

        def shaded(f):
            one, two = f.materials[0], f.materials[1]
            one.color          = np.array([0.25, 0.5, 0.75])
            one.diffuse        = 0.9
            one.ambient_color  = np.array([0.3, 0.2, 0.1])
            one.incandescence  = np.array([0.0, 0.4, 0.6])
            one.transparency   = np.array([0.5, 0.5, 0.25])
            two.color          = np.array([0.1, 0.1, 0.9])
            two.specular_color = np.array([0.9, 0.8, 0.7])
            two.cosine_power   = 12.5
            two.reflectivity   = 0.75
            two.incandescence  = np.array([0.2, 0.2, 0.2])

        def blinn(f):
            # Maya's blinn reads as a phong
            f.materials[2].specular_color = np.array([0.1, 0.2, 0.3])
            f.materials[2].reflectivity   = 0.6
            f.materials[2].transparency   = np.array([0.2, 0.2, 0.2])

        def wraps(f):
            f.materials[0].wrap_u = "clamp"
            f.materials[0].wrap_v = "repeat"

        def uv_set(f):
            f.uvs[0].name         = "other"
            f.materials[0].uv_set = "other"

        def a_face(f):
            one, two = f.materials[0], f.materials[1]
            face      = int(one.faces[0])
            one.faces = np.asarray([x for x in one.faces if x != face])
            two.faces = np.sort(np.append(two.faces, face))

        def uuid(f):
            f.materials[2].uuid = "ABCD" + f.materials[2].uuid[4:]

        def kind(f):
            f.materials[0].kind = "phong"

        def texture(f):
            f.materials[0].texture = 255 - np.asarray(f.materials[0].texture)

        def normal_map(f):
            f.materials[1].normal_map = np.full((2, 2, 3), 128, dtype=np.uint8)

        def mesh(f):
            f.materials[2].mesh_uuid = f.meshes[0].uuid

        def some_faces(f):
            f.materials[2].faces = np.array([0, 1, 2])

        def wrap_of_no_map(f):
            f.materials[1].wrap_u = "clamp"

        self.assertEach(
            saves=[
                (every, shaded),
                (embedded, shaded),
                (embedded, blinn),
                (every, wraps),
                (every, uv_set),
                (every, a_face),
                (every, uuid),
            ],
            refusals=[
                (every, kind, "keeps its kind"),
                (every, texture, "keeps the maps"),
                (every, normal_map, "keeps the maps"),
                (every, mesh, "names another mesh than the file's"),
                (every, some_faces, "a mesh is covered whole or not at all"),
                (every, wrap_of_no_map, "materials[1].wrap_u would read back otherwise"),
            ],
        )
        self.assertSaves(linked, shaded, name="linked.fbx")
        self.assertEqual(loaded(self.path("uv_set.fbx")).materials[0].uv_set, "other")

    def test_names(self):
        embedded = maya_file("materials_export", "maya_embedded.fbx")

        def renamed(f):
            f.materials[0].name = "skin"
            f.materials[2].name = "box"

        def bare_renamed(f):
            # a material of another shading model: a name and faces alone
            f.materials[3].name = "standard"

        def bare_colour(f):
            f.materials[3].color = np.array([0.9, 0.1, 0.1])

        self.assertEach(
            saves    = [(SHARED["every"], renamed), (embedded, bare_renamed)],
            refusals = [(embedded, bare_colour, "neither a lambert")],
        )
        names = [material.name for material in loaded(self.path("renamed.fbx")).materials]
        self.assertEqual(names, ["skin", "shinyPhong", "box"])
        self.assertEqual(loaded(self.path("bare_renamed.fbx")).materials[3].name, "standard")

    def test_normal_map(self):
        def mapped(held):
            first                 = held.materials[0]
            first.normal_map      = np.full((4, 4, 3), 128, dtype=np.uint8)
            first.normal_map_kind = "tangent"
            first.bump_depth      = 0.5

        def depth(f):
            f.materials[0].bump_depth = 0.25
            f.materials[0].wrap_u     = "clamp"

        self.assertSaves(self.built("mapped.fbx", SHARED["every"], mapped), depth)
        got = loaded(self.path("saved.fbx")).materials[0]
        self.assertEqual((got.bump_depth, got.wrap_u, got.normal_map_kind), (0.25, "clamp", "tangent"))

    def test_one_material_on_two_meshes(self):
        """the records of one uuid are one material of the file: changed
        alike they save, one alone is refused"""

        def twinned(held):
            twin              = held.materials[2].copy()
            twin.mesh_uuid    = held.materials[3].mesh_uuid
            twin.uuid         = held.materials[2].uuid
            held.materials[3] = twin

        def alike(f):
            self.assertEqual(f.materials[2].uuid, f.materials[3].uuid)
            for record in (f.materials[2], f.materials[3]):
                record.color = np.array([0.4, 0.3, 0.2])
                record.name  = "both"

        def alone(f):
            f.materials[3].color = np.array([0.4, 0.3, 0.2])

        def apart(f):
            f.materials[3].uuid  = "ABCD" + f.materials[0].uuid[4:]
            f.materials[3].color = np.array([0.4, 0.3, 0.2])

        source = self.built("twinned.fbx", maya_file("materials_export", "maya_embedded.fbx"), twinned)
        self.assertEach(
            saves    = [(source, alike)],
            refusals = [(source, alone, "one material is written per uuid"), (source, apart, "no longer agree")],
        )


@unittest.skipIf(FBX is None, "Autodesk FBX Python SDK is not installed")
class TestScenePutBack(_Edits):
    """a save that fails leaves the handle as it was: the records keep
    their changes, the open scene is the file again, a later save reads as
    the records"""

    @staticmethod
    def shifted(f) -> np.ndarray:
        """the second node of the first take moved along x"""
        take      = f.clip[0]
        translate = np.array(take.frames.translate)
        translate[:, 1, 0] += 5.0
        take.frames.translate = translate
        return translate

    def test_unwritable(self):
        """a path under a file cannot be written: the records keep the
        change and the next save writes it"""
        blocker = self.path("blocker")
        with open(blocker, "w"):
            pass
        with FbxFile(SHARED["every"], verbose=False) as f:
            kept = self.shifted(f)
            with self.assertRaises(OSError):
                f.save(os.path.join(blocker, "x.fbx"))
            np.testing.assert_array_equal(np.asarray(f.clip[0].frames.translate), kept)
            f.save(self.path("after.fbx"))
            self.assertNear(loaded(self.path("after.fbx")), everything(f), "after: ")
            # a kind first read after the failure is the file's
            self.assertEqual(list(f.materials), list(loaded(SHARED["every"]).materials))
        self.assertEqual(os.path.getsize(blocker), 0)

    def test_refused_then_saved(self):
        """a changed save, a refused one, the value put back: the handle
        saves again, as its records"""
        with FbxFile(SHARED["every"], verbose=False) as f:
            self.shifted(f)
            f.save(self.path("one.fbx"))
            node           = f.hierarchy.list[1]
            held           = np.array(node.translate)
            node.translate = [9.0, 9.0, 9.0]
            with self.assertRaises(ValueError) as caught:
                f.save(self.path("no.fbx"))
            self.assertIn("is given another translate in the hierarchy of a file holding takes", str(caught.exception))
            self.assertFalse(os.path.exists(self.path("no.fbx")))
            node.translate = held
            f.save(self.path("two.fbx"))
            self.assertNear(loaded(self.path("two.fbx")), everything(f), "two: ")
        same_read(self, loaded(self.path("two.fbx")), loaded(self.path("one.fbx")), "two as one: ")

    def test_refused_part_way(self):
        """two kinds changed, the second refused after the first was
        written into the scene: both put back, the file saves as it was
        read"""
        source = SHARED["every"]
        with FbxFile(source, verbose=False) as f:
            f.save(self.path("plain.fbx"))
        with FbxFile(source, verbose=False) as f:
            mesh, camera = f.meshes[0], f.cameras[0]
            points, scale = np.array(mesh.points), camera.camera_scale
            mesh.points         = points + 1.0
            camera.camera_scale = 2.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaises(ValueError) as caught:
                    f.save(self.path("no.fbx"))
            self.assertIn(
                "the change of meshes and cameras cannot be written into this file: "
                "cameras[0].camera_scale would read back otherwise",
                str(caught.exception),
            )
            mesh.points         = points
            camera.camera_scale = scale
            f.save(self.path("back.fbx"))
        self.assertEqual(census(self.path("back.fbx")), census(self.path("plain.fbx")))
        same_read(self, loaded(self.path("back.fbx")), loaded(source), "back: ")

    def test_saved_twice(self):
        """a changed save, then a save of nothing more: both files read
        the same, and a kind first read after the save is the saved
        file's"""
        with FbxFile(SHARED["every"], verbose=False) as f:
            f.clip[0].take = "other"
            self.shifted(f)
            f.save(self.path("one.fbx"))
            f.save(self.path("two.fbx"))
            self.assertEqual(list(f.materials), list(loaded(self.path("one.fbx")).materials))
        same_read(self, loaded(self.path("two.fbx")), loaded(self.path("one.fbx")), "two as one: ")
        self.assertEqual(census(self.path("two.fbx")), census(self.path("one.fbx")))


if __name__ == "__main__":
    unittest.main()
