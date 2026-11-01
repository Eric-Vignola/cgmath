"""load_fbx against Maya's FBXImport.

test_assets/fbx_maya_import.npz holds where Maya 2025 put every leaf and every
node stored eInheritRrSs or eInheritRrs on 180 files, on every whole frame and
with animation off, recorded by tools/make_fbx_maya_import_fixture.py (its
docstring lists the arrays and why the files sit on a grid single precision
holds). No fbx file ships: a file is written with the sdk, from the layout the
archive holds, the first time a test reads it (_fbx_import_files.py). Nothing
here needs Maya, but all of it needs the sdk, so it skips under plain python.

Each file has a limit, in the archive, to the bit: the largest position gap
over its clip and its rig. The ceilings: 2e-14 in position, 5e-14 on
s0_s2m1h_Gt, 1e-13 on ro_dc_G123_gk and xup_ns_ksc1_Gt123_gs, each past 2e-14 by
Maya's rounding, and s0_q38's 3x3 at 4.440892098500626e-15. Maya's own single
precision is not copied, so a file of free values sits up to about 1e-6 off.
112 more files, the archive's "edges", each pin a rule those 180 cannot see,
on values single precision does not hold, within 2e-6.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from cgmath.hierarchy import ClipData, HierarchyData, TransformData, TransformList

try:
    import fbx as FBX

    from cgmath._tests._fbx_import_files import add_curve, write_layout
except ImportError:
    FBX = None

H = sys.modules[ClipData.__module__]

FIXTURE  = os.path.join(os.path.dirname(__file__), "test_assets", "fbx_maya_import.npz")

CEILING  = 2e-14
CEILINGS = {"s0_s2m1h_Gt": 5e-14, "ro_dc_G123_gk": 1e-13, "xup_ns_ksc1_Gt123_gs": 1e-13}
Q38_AXES = 4.440892098500626e-15

# the shear no joint holds, a joint taking what xform -m gives one, its worst
# over the kept kinds on these files: a joint in position, in the 3x3 up to this
JOINT_AXES = 14.77

KINDS = (("joint",), ("joint", "transform"), ("joint", "locator"), ("transform",), ("locator",))

# the edge files, each a rule the 180 cannot see, sit off the grid single
# precision holds where that rule needs them: Maya's own rounding, a(f)'s
EDGE = 2e-6

# the reads past their ceiling, the fold's rounding on nodes no read covers
# better: every one a few times 1e-14
LEFT_OUT = {
    ("joint",): ("gm_s222_Gt", "gm_ksceven", "ns_ksceven", "ns_amr_G123_gs", "ev_h8"),
    ("joint", "locator"): ("xup_gm_s222_G3", "ns_amr_G123_gs"),
}

# parents Maya's FBXImport classified, a fresh import each: (S1, R1, S2, R2,
# whether an eInheritRrSs node under S1 R1 S2 R2 baked), the 40 of 2,100 that
# sit nearest the band and that tell the ways to get it wrong apart
PARENTS = (
    ((0.49986166989927006, 0.5000197470642855, 0.4999055341847748), (0, 75, 120), (0.9999028516356371, 1.0001489099346093, 1.000233029729059), (90, 120, 113.3879505498727), True),
    ((0.9998886312662592, 1.00008087702638, 1.0000160460333556), (30, 0, 90), (1.0000609314716973, 1.000117908805699, 0.999926599640979), (0, 15, 90), True),
    ((0.4999952698110976, 0.500024704289349, 0.5000744738390304), (75, -30, 120), (0.9998241488960712, 0.9999266401092042, 0.9999032974922556), (120, -30, -60), False),
    ((0.4999991066121977, 0.500044505476456, 0.500036023952623), (30, 30, 45), (1.0000287808324124, 1.0000453166028107, 0.9999473425604729), (90, 120, 60), False),
    ((1.9998363059351494, 1.9998591771485787, 1.9998253505638974), (0, 15, 0), (0.9999077152947787, 1.0, 0.999968358445143), (90, 45, 45), False),
    ((0.49996846061898975, 0.5, 0.49990377905175853), (15, -60, 90), (0.9999296534373063, 1.0, 0.9998595738956945), (30, -60, -62.254459452444436), True),
    ((2.0000342910970583, 2.0001309700586574, 1.9998335041110982), (-30, 0, 30), (0.999994761684171, 1.000032038632244, 1.0), (45, 90, -30), False),
    ((0.5000275881935825, 0.500042137558233, 0.5), (30, -52.91300281080083, 0), (0.9999613634208115, 0.9997687039922123, 0.9997165827920323), (45, 75, 0), True),
    ((1.0, 1.0001096261117222, 1.0), (-60, -30, 75), (0.999888987196589, 1.0, 1.0), (0, 90, 60), False),
    ((4.000095607966589, 3.999974090777372, 4.0), (75, 90, 0), (0.9999849956743975, 1.0000374983293074, 1.0), (15, 0, 45), False),
    ((-10.001000458250092, 9.999217062798893, 9.999456670752666), (30, 90, 30), (1.0000344106849872, 0.9999693363643611, 0.9998916265882463), (90, 0, 45), True),
    ((-0.5000130992821435, -0.49995767354877724, -0.5), (-143.36538146744576, 15, 90), (1.0000806006813412, 1.0000459082909612, 1.0), (45, 45, -60), True),
    ((0.2499966431673801, -0.25, -0.25), (15, 75, 15), (0.9999968906641614, 0.9999266980733735, 1.000044801626989), (45, 0, 45), True),
    ((0.25001041290969933, 0.25, 0.2499971115641204), (-30, -30, 15), (1.0000045058133316, 0.9999287891300939, 1.0), (-60, -60, -89.861757172688), False),
    ((0.25001160868182365, 0.25, 0.25), (90, 155.92787092935623, 120), (1.0001902418985549, 1.0, 0.9999718029104933), (120, 0, 120), False),
    ((0.9999852288644806, 1.0000518531665967, 0.9999809643470232), (120, 15, -20.139521975767195), (1.0000517580548578, 1.0000256489968227, 0.9999613446086437), (60, 30, 45), False),
    ((1.0, 1.0000777499044833, 1.0), (-60, -30, 75), (1.0, 0.9998908810971129, 0.9999277652018869), (-30, 75, 45), False),
    ((-0.250013132052568, 0.25, 0.25002196609286836), (0, 120, 0), (0.9999313290843291, 0.9999684981358711, 1.0000456587497932), (-30, 45, 30), True),
    ((0.2500027070342678, 0.24997801632500502, 0.24999309231529498), (0, 75, 45), (0.9999236170183157, 1.000076900092318, 1.0), (-30, 75, -30), False),
    ((4.000092603226568, 4.000129986596185, 3.99994258193347), (15, 114.23661505425906, 30), (1.000042124974747, 1.0, 1.0000104124141953), (90, 115.67700646534354, 0), True),
    ((4.000170432811478, -3.999801051210188, -4.0), (37.320976867001434, -149.0835425830184, -30), (0.9999997547796178, 0.9999866699865951, 1.0000189774851218), (90, 15, 45), False),
    ((0.5, 0.5000265016977397, 0.49996659603577287), (0, -60, 60), (0.9999809192734697, 1.000026755150092, 0.9999223534566745), (75, 75, 45), False),
    ((-0.9999419639504258, 1.0000513571047653, 1.0000965405232354), (60, 0, -30), (1.0000671791766884, 0.9999308465871846, 1.0000230162825887), (45, -38.1960959731407, -60), True),
    ((0.25002451003342, 0.2500089123136398, 0.24999268883255174), (-30, -152.05850635600967, 45), (1.000095196664883, 0.9999381110575825, 1.0), (15, 120, 90), True),
    ((1.0, 1.0, 0.9999891997317448), (0, 0, 120), (1.0, 0.9999217453172733, 1.00005216855148), (120, 78.89813837563872, 15), True),
    ((-2.0, -1.9999380788084875, -2.0), (121.89634178169888, 60, -60), (0.9999498194964849, 1.0, 1.0000557240207835), (120, -30, 120), False),
    ((0.9999683614080512, 1.0, 0.9999235475034377), (0, -30, 15), (1.0, 1.0, 1.0000882688352022), (60, -30, -126.65244550624683), True),
    ((2.0001731856163034, 2.0000188515179835, 2.0001914506581953), (45, -60, -30), (1.0, 1.0, 1.000039970865097), (-30, 45, 90), True),
    ((0.5, -0.49997792494346793, 0.5000287749596874), (15, 15, 45), (0.9999324529254613, 0.9999660379394665, 1.0000063708312847), (30, 45, 60), True),
    ((4.0, -4.0001409341314265, -3.999976593091764), (120, 60, 0), (0.9999656554894624, 1.0, 1.0), (90, 0, 60), True),
    ((0.9999562170025104, -1.0000168809703436, 0.9999838923977709), (0, 75, 0), (0.9999713503845261, 1.0000131991107584, 0.9999716752037979), (-30, 0, 0), True),
    ((-0.5000310903670823, -0.5, -0.5000026655513147), (60, 30, 75), (0.9999405014622227, 1.0, 0.9999805475749113), (0, 0, 90), True),
    ((-0.49997402920852857, -0.4999083487355639, -0.5), (162.0537715755724, 0, -60), (1.0, 0.9999288938723749, 0.9999336009248704), (32.31573941819016, -108.52025462378589, -30), True),
    ((1.999924783060816, 2.0000338828192072, 2.0000336107395023), (120, 75, -60), (1.0000118177030999, 1.0000497337847434, 1.0), (-60, -60, 45.50889602387457), False),
    ((1.000099022324729, -1.0000359979302185, 0.9999610228147665), (0, -60, -30), (1.0000238315940986, 1.0000999880851245, 0.9999668595130904), (120, 45, -144.07311124472872), False),
    ((0.24998284708741778, -0.25, 0.25), (45, 60, 0), (1.0001001208060096, 0.9998438444364529, 1.000161074244434), (45, -30, 90), False),
    ((1.0000095943144673, 0.9999319285615215, 1.0), (30, 60, 45), (1.0, 0.9998283672332144, 0.9998380817492344), (90, 90, 30), False),
    ((0.5000460154176788, -0.500066234416229, 0.5), (-30, -60, 90), (1.0, 0.9999689444873275, 1.0), (120, 120, 30), True),
    ((-10.0, -9.998970315710563, -9.999029830137735), (0, 0, 0), (0.9998148681035499, 1.00001575338721, 0.9998850619005201), (-28.586671150343165, 90, 45), False),
    ((3.999856205267533, 4.0, 4.000354433989969), (15, 0, 0), (1.0, 1.0, 0.9999724209912648), (45, 75, 15), False),
)

_CACHE = {}

# what the reads log stays off the console while the module runs: the logger
# keeps a handler of its own in place of its parents', and logged() adds one
# beside it
_QUIET = logging.NullHandler()


def setUpModule():
    logger              = logging.getLogger("cgmath.formats.fbx")
    _CACHE["propagate"] = logger.propagate
    logger.addHandler(_QUIET)
    logger.propagate = False


def tearDownModule():
    logger = logging.getLogger("cgmath.formats.fbx")
    logger.removeHandler(_QUIET)
    logger.propagate = _CACHE.pop("propagate", True)
    folder           = _CACHE.pop("folder", None)
    if folder is not None:
        folder.cleanup()
    _CACHE.clear()


def records() -> dict:
    """the archive, read once"""
    if "data" not in _CACHE:
        with np.load(FIXTURE) as data:
            _CACHE["data"] = {key: data[key] for key in data.files}
    return _CACHE["data"]


def layout_of(data: dict, name: str) -> dict:
    return json.loads(bytes(data[name + "|layout"]))


def fbx_path(name: str) -> str:
    """the file ``name`` of the archive, written the first time it is asked for"""
    if "folder" not in _CACHE:
        _CACHE["folder"] = tempfile.TemporaryDirectory()
    path = os.path.join(_CACHE["folder"].name, name + ".fbx")
    if not os.path.exists(path):
        write_layout(path, layout_of(records(), name))
    return path


def whole_frames(data: dict, name: str) -> np.ndarray:
    """the rows of the record that sit on whole frames"""
    frames = data[name + "|frames"]
    return np.flatnonzero(frames == np.floor(frames))


def blocks(worlds) -> np.ndarray:
    """(..., 12) the 4 x 3 of (..., 4, 4) worlds, as the archive holds them"""
    worlds = np.asarray(worlds)
    return worlds[..., :3].reshape(worlds.shape[:-2] + (12,))


def gap(got, want) -> tuple:
    """(3x3, position), the largest gaps between (..., 12) records"""
    delta = np.abs(np.asarray(got) - want)
    return float(delta[..., :9].max()), float(delta[..., 9:].max())


def read_gaps(data: dict, name: str, path: str, kinds=None) -> tuple:
    """({node: gap} of the clip on every whole frame, the same of the rig
    against the record with animation off), the nodes the archive records"""
    frames = data[name + "|frames"]
    whole  = whole_frames(data, name)
    clip   = ClipData.load_fbx(path, start_frame=int(frames[whole[0]]), end_frame=int(frames[whole[-1]]), node_types=kinds)
    rig    = HierarchyData.load_fbx(path, node_types=kinds)
    worlds = blocks(clip.frames.world_matrix)
    frame  = {}
    for column, node in enumerate(clip.name):
        if f"{name}|{node}" in data:
            frame[node] = gap(worlds[:, column], data[f"{name}|{node}"][whole])
    rest = {
        node.name: gap(blocks(node.world_matrix), data[f"{name}|s|{node.name}"])
        for node in rig
        if f"{name}|s|{node.name}" in data
    }
    return frame, rest


def position_gaps(data: dict, name: str, folder: str) -> float:
    """the file's limit, the tool's --limits"""
    path = os.path.join(folder, name + ".fbx")
    write_layout(path, layout_of(data, name))
    return max(max(v[1] for v in gaps.values()) for gaps in read_gaps(data, name, path))


def matrix(sdk_matrix) -> np.ndarray:
    return np.array([[sdk_matrix.Get(row, column) for column in range(4)] for row in range(4)])


def frame_time(frame: float, fps: float = 24.0):
    time = FBX.FbxTime()
    time.SetSecondDouble(frame / fps)
    return time


def spec(name, kind, parent, **over) -> dict:
    """a node dict of a layout (_fbx_import_files.py)"""
    out = dict(
        name=name, kind=kind, parent=parent, t=(0, 0, 0), r=(0, 0, 0), s=(1, 1, 1), order="xyz", pre=None, post=None,
        inherit="Rrs" if kind == "joint" else "RSrs", piv=None, keys={}, ckeys=[], points=None,
    )
    out.update(over)
    return out


def written(folder, name, nodes, **over) -> str:
    path = os.path.join(folder, name + ".fbx")
    write_layout(path, dict(dict(nodes=nodes, axes="yup", span=(0, 6)), **over))
    return path


class _Logged(logging.Handler):
    """the messages logged at WARNING and above"""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def logged(read, path) -> list:
    """the warnings ``read(path)`` logs on cgmath.formats.fbx's logger"""
    logger  = logging.getLogger("cgmath.formats.fbx")
    handler = _Logged()
    logger.addHandler(handler)
    try:
        read(path)
    finally:
        logger.removeHandler(handler)
    return handler.messages


@unittest.skipIf(FBX is None, "the Autodesk FBX SDK is required")
class TestMayaImport(unittest.TestCase):
    """every file, every read, against Maya's record"""

    @classmethod
    def setUpClass(cls):
        cls.data  = records()
        cls.names = cls.data["names"].tolist()

    def limit(self, name: str) -> float:
        return float(self.data["limits"][self.names.index(name)])

    def test_every_file(self):
        """each of the 180: every whole frame of a clip against Maya's
        record, the rig against the record with animation off: position
        within the file's limit, itself within its ceiling, the 3x3 within
        1e-9, s0_q38's within its own"""
        self.assertEqual(len(self.names), 180)
        for name in self.names:
            limit = self.limit(name)
            with self.subTest(name):
                self.assertLessEqual(limit, CEILINGS.get(name, CEILING))
                frame, rest = read_gaps(self.data, name, fbx_path(name))
                self.assertTrue(frame and rest)
                for read, gaps in (("clip", frame), ("rig", rest)):
                    with self.subTest(name, read=read):
                        axes     = max(value[0] for value in gaps.values())
                        position = max(value[1] for value in gaps.values())
                        self.assertLessEqual(axes, Q38_AXES if name == "s0_q38" else 1e-9)
                        self.assertLessEqual(position, limit)

    def test_the_edges(self):
        """each edge file against Maya's record, read whole and joints alone:
        a joint in position, any other node in position and 3x3. The even test
        on the static pose alone, a scale curve above the parent, a dropped
        node of identity channels, the conversion's own translate keys under
        two live layers; a skinned mesh's static bake; a bake to the later of
        0 and its last key; a last key off a whole frame on a folded node, and
        keys the conversion leaves as they are; a node with no attribute under
        a joint, its pivots, its loose keys, its rotation inactive; a joint
        root's pivots in a turned file; a keyed node through the gimbal of its
        order; a folded node re-keyed from the whole frame its first key
        before 0 truncates to; a node keyed in a later layer, muted or not,
        channel by channel, the conversion run on the first layer alone, a
        live layer over a first-layer curve it empties counted once, below a
        head the first layer held under the live layers, on rotate and on
        scale, a merged channel held outside the take's span; a skinned mesh
        under a camera root of a turned file, the camera mirrored, baked,
        keyed, the mesh keyed, on one axis too, with a pre or a post rotation,
        in another order, under a muted later layer, on Maya's other euler
        solution, at a half turn, and with pivots, in rotate and in scale, one
        with no skin; a camera under a joint; an emptied axis keyed before 0"""
        names = self.data["edges"].tolist()
        self.assertEqual(len(names), 112)
        for name in names:
            kinds = {node["name"]: node["kind"] for node in layout_of(self.data, name)["nodes"]}
            with self.subTest(name):
                for read in (None, ("joint",)) if "joint" in kinds.values() else (None,):
                    frame, rest = read_gaps(self.data, name, fbx_path(name), read)
                    self.assertTrue(read or (frame and rest))
                    for gaps in (frame, rest):
                        for node, (axes, position) in gaps.items():
                            self.assertLessEqual(position, EDGE, (read, node))
                            if kinds[node] != "joint":
                                self.assertLessEqual(axes, EDGE, (read, node))

    def test_a_sheared_camera_warns(self):
        """a read, rig, view or clip, logs one warning naming each camera
        scaled unevenly or sheared, its own three scale magnitudes apart or
        its world sheared by its parent, by more than the band relative to
        their size, at the static pose or on a frame the read takes, a clip
        both, and says a later release refuses such a file; an evenly scaled
        camera, a mirrored one, its three scale magnitudes equal, and one
        within the band warn of nothing"""
        reads = {
            "rig":  HierarchyData.load_fbx,
            "view": TransformList.load_fbx,
            "clip": lambda path: ClipData.load_fbx(path, start_frame=0, end_frame=8),
        }
        with tempfile.TemporaryDirectory() as folder:
            files = {
                # e_camera_skin's camera C is scaled (1, 2, 3), e_camera_skin_typed's 1
                "uneven":   fbx_path("e_camera_skin"),
                "even":     fbx_path("e_camera_skin_typed"),
                "mirrored": written(folder, "mirrored", [spec("C", "camera", None, r=(10, 30, -20), s=(-2, 2, 2)), spec("l", "locator", "C")]),
                # A sheared by its parent, B scaled unevenly, E evenly: one
                # warning names A and B
                "two": written(
                    folder,
                    "two",
                    [
                        spec("P", "group", None, r=(0, 30, 0), s=(1, 2, 3)),
                        spec("A", "camera", "P", r=(20, 40, 60)),
                        spec("B", "camera", None, s=(1, 1, 0.5)),
                        spec("E", "camera", None, r=(20, 40, 60), s=(2, 2, 2)),
                        spec("l", "locator", "A"),
                    ],
                ),
                # even at the static pose, uneven on frame 4
                "keyed": written(
                    folder,
                    "keyed",
                    [spec("C", "camera", None, keys={"s": ([0, 4, 8], [(1, 1, 1), (1, 2, 1), (1, 1, 1)])}), spec("l", "locator", "C")],
                    span=(0, 8),
                ),
                # either side of the band: U named, Q quiet
                "band": written(
                    folder,
                    "band",
                    [spec("U", "camera", None, s=(1, 1, 1.0002)), spec("Q", "camera", None, s=(1, 1, 1.0001)), spec("l", "locator", "U")],
                ),
                # 5e-5 apart relative to its size, 5e-4 as it stands: quiet
                "large": written(folder, "large", [spec("C", "camera", None, s=(10, 10, 10.0005)), spec("l", "locator", "C")]),
                # uneven at the static pose, even on every frame: a clip names
                # it too
                "rest": written(
                    folder,
                    "rest",
                    [spec("C", "camera", None, s=(1, 2, 1), keys={"s": ([0, 8], [(1, 1, 1), (1, 1, 1)])}), spec("l", "locator", "C")],
                    span=(0, 8),
                ),
            }
            named = {"uneven": "C", "two": "A, B", "band": "U", "rest": "C"}
            for read, call in reads.items():
                for kind, path in files.items():
                    with self.subTest(read=read, file=kind):
                        messages = logged(call, path)
                        name     = "C" if kind == "keyed" and read == "clip" else named.get(kind)
                        if name is None:
                            self.assertEqual(messages, [])
                            continue
                        self.assertEqual(len(messages), 1, messages)
                        self.assertTrue(messages[0].endswith(": " + name), messages[0])
                        self.assertIn("a later release will refuse", messages[0])

    def test_node_types(self):
        """a read of each kept kind places a kept non-joint in position and
        3x3 and a kept joint in position, within the same ceilings, a joint's
        axes off by the shear it can't hold and no more, and every joint
        holds no shear"""
        for kinds in KINDS:
            keeps = set(kinds) | ({"group"} if "transform" in kinds else set())
            for name in self.names:
                nodes = {node["name"]: node["kind"] for node in layout_of(self.data, name)["nodes"]}
                if not keeps & set(nodes.values()):
                    continue
                with self.subTest(name, kinds=kinds):
                    path = fbx_path(name)
                    frame, rest = read_gaps(self.data, name, path, kinds)
                    ceiling = CEILINGS.get(name, CEILING)
                    for gaps in (frame, rest):
                        for node, (axes, position) in gaps.items():
                            if nodes[node] == "joint":
                                self.assertLessEqual(axes, JOINT_AXES, node)
                            if name in LEFT_OUT.get(kinds, ()):
                                continue
                            self.assertLessEqual(position, ceiling, node)
                            if nodes[node] != "joint":
                                self.assertLessEqual(axes, 1e-9, node)
                    rig  = HierarchyData.load_fbx(path, node_types=kinds)
                    clip = ClipData.load_fbx(path, node_types=kinds)
                    for index, node in enumerate(rig):
                        if node.node_type == "joint":
                            self.assertFalse(np.any(node.shear), node.name)
                            self.assertFalse(np.any(clip.frames.shear[:, index]), node.name)

    def test_channels(self):
        """a joint keeps the file's rotate and scale where nothing dropped
        moves it and its flag as stored; a baked node keeps its rotate order
        and holds no rotate axis and no joint orient; a node holds shear
        where Maya's record does; a clip frames it"""
        for name, count in (("s0_s123", 2), ("s0_s222", 0), ("x_kp_xyz_e", 0)):
            with self.subTest(name):
                rig    = HierarchyData.load_fbx(fbx_path(name))
                frames = self.data[name + "|frames"]
                clip   = ClipData.load_fbx(fbx_path(name), start_frame=int(frames[0]), end_frame=int(frames[-1]))
                total  = 0
                for node in rig:
                    key = f"{name}|h|{node.name}"
                    if key not in self.data or node.node_type == "joint":
                        continue
                    maya = self.data[key]
                    held = clip.frames.shear[:, clip.index(node.name)]
                    self.assertEqual(held.shape, (clip.frame_count, 3))
                    self.assertLessEqual(float(np.abs(held - maya).max()), 1e-6, node.name)
                    total += bool(np.abs(maya).max() > 1e-9)
                self.assertEqual(total, count)

        path  = fbx_path("x_jall_s123_gs")
        rig   = HierarchyData.load_fbx(path)
        nodes = {node["name"]: node for node in layout_of(self.data, "x_jall_s123_gs")["nodes"]}
        for node in rig:
            stored = nodes[node.name]
            if node.node_type == "joint":
                self.assertEqual(node.segment_scale_compensate, stored["inherit"] == "Rrs", node.name)
                self.assertTrue(np.array_equal(node.rotate, stored["r"]), node.name)
                self.assertTrue(np.array_equal(node.scale, stored["s"]), node.name)
            else:
                self.assertFalse(node.segment_scale_compensate, node.name)
        for name in ("gRrs", "gRrSs"):
            order = ("xyz", "yzx", "zxy", "xzy", "yxz", "zyx").index(nodes[name]["order"])
            self.assertNotEqual(order, 0)
            self.assertEqual(rig[name].rotate_order, order, name)
            self.assertFalse(np.any(rig[name].rotate_axis), name)
            self.assertFalse(np.any(rig[name].joint_orient), name)
            self.assertTrue(np.any(rig[name].shear), name)

    def test_half_frames(self):
        """a read at twice the file's rate on Maya's 13 samples: a baked node
        steps as Maya's keys do, an unbaked one blends"""
        for name in ("x_sp_s123", "x_sl_end14", "x_pv_s123", "x_pv_dc_G123_gs"):
            with self.subTest(name):
                frames = self.data[name + "|frames"]
                self.assertEqual(len(frames), 13)
                clip   = ClipData.load_fbx(fbx_path(name), start_frame=0, end_frame=12, fps=48.0)
                worlds = blocks(clip.frames.world_matrix)
                count  = 0
                for column, node in enumerate(clip.name):
                    key = f"{name}|{node}"
                    if key in self.data:
                        count += 1
                        self.assertLessEqual(max(gap(worlds[:, column], self.data[key])), 1e-13, node)
                self.assertGreater(count, 10)

    def test_file_axes(self):
        """axis_system=None: the rule runs the same and the turn stays off,
        a root that bakes included"""
        for name in ("s0_zup_q38", "xup_x_kp_yzx", "zup_tr_rrs", "xup_tr_rrs_m"):
            with self.subTest(name):
                path = fbx_path(name)
                turn = np.eye(4)
                with H.open_fbx(path, FBX) as scene:
                    turn[:3, :3] = H._fbx_axis_turn(scene, "maya_y_up", path)
                for read in (ClipData, HierarchyData):
                    own  = read.load_fbx(path, axis_system=None)
                    want = read.load_fbx(path)
                    get  = (lambda data: data.frames.world_matrix) if read is ClipData else (lambda data: data.world_matrix)
                    self.assertLessEqual(float(np.abs(np.asarray(get(own)) @ turn - np.asarray(get(want))).max()), 1e-13)
                rig = HierarchyData.load_fbx(path, axis_system=None)
                for node in rig:
                    key = f"{name}|s|{node.name}"
                    if key in self.data:
                        world = blocks(np.asarray(node.world_matrix) @ turn)
                        self.assertLessEqual(max(gap(world, self.data[key])), 1e-13, node.name)

    def test_a_window(self):
        """a window of frames reads what the full read gives them: frames
        past a last key, a start_frame past every key, a mixed node's solve"""
        def windows(name, spans, first, last):
            path = fbx_path(name)
            full = np.asarray(ClipData.load_fbx(path, start_frame=first, end_frame=last).frames.world_matrix)
            rows = {int(frame): row for row, frame in enumerate(self.data[name + "|frames"]) if frame == int(frame)}
            for start, end in spans:
                with self.subTest(name, start=start, end=end):
                    part   = ClipData.load_fbx(path, start_frame=start, end_frame=end)
                    worlds = np.asarray(part.frames.world_matrix)
                    self.assertLessEqual(float(np.abs(worlds - full[start - first : end - first + 1]).max()), 1e-13)
                    known = [frame for frame in range(start, end + 1) if frame in rows]
                    for column, node in enumerate(part.name):
                        key = f"{name}|{node}"
                        if known and key in self.data:
                            want = self.data[key][[rows[frame] for frame in known]]
                            got  = blocks(worlds[[frame - start for frame in known], column])
                            self.assertLessEqual(max(gap(got, want)), 1e-13, node)

        windows("x_sl_end14", ((5, 6), (3, 5)), 0, 6)
        windows("x_far", ((1008, 1009), (1003, 1005)), 998, 1009)
        for name in ("x_kp_zxy_m", "x_mx_zyx", "x_mx_q"):
            windows(name, ((4, 8), (5, 8), (7, 8)), 0, 8)

        # a parent even at the static pose alone: the even test reads no
        # frame, so no window of it bakes, nor does the full read
        windows("e_even_mirror", ((1, 2), (3, 5)), 0, 6)

    def test_rest_is_the_rig(self):
        """a clip's rest pose and a TransformList read are the rig, to the bit"""
        for name in self.names:
            with self.subTest(name):
                path = fbx_path(name)
                rig  = np.asarray(HierarchyData.load_fbx(path).world_matrix)
                self.assertTrue(np.array_equal(np.asarray(ClipData.load_fbx(path).rest_pose.world_matrix), rig))
                view = HierarchyData(TransformList.load_fbx(path))
                self.assertTrue(np.array_equal(np.asarray(view.world_matrix), rig))

    def test_points(self):
        """no bake reaches a mesh's points, and under the rig's world they
        are where Maya has them"""
        from cgmath.geometry.mesh import load_fbx as mesh_load_fbx

        for name in [name for name in self.names if "x_me_" in name]:
            with self.subTest(name):
                path = fbx_path(name)
                rig  = HierarchyData.load_fbx(path)
                done = 0
                for mesh, _ in mesh_load_fbx(path):
                    maya   = self.data[f"{name}|p|{mesh.name}"]
                    points = np.asarray(mesh.points)
                    self.assertTrue(np.array_equal(points, maya), mesh.name)
                    world = np.asarray(rig[mesh.name].world_matrix)
                    want  = self.data[f"{name}|s|{mesh.name}"].reshape(4, 3)
                    got   = points @ world[:3, :3] + world[3, :3]
                    self.assertLessEqual(float(np.abs(got - (maya @ want[:3] + want[3])).max()), self.limit(name), mesh.name)
                    done += 1
                self.assertGreaterEqual(done, 8)

        # numbers single precision does not hold stay the file's
        free = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.2), (0.3, 0.7, -0.1), (-0.4, 0.5, 0.6)]
        with tempfile.TemporaryDirectory() as folder:
            path = written(folder, "free", [spec("M", "mesh", None, points=free)])
            mesh = mesh_load_fbx(path)[0][0]
            self.assertTrue(np.array_equal(np.asarray(mesh.points), free))


@unittest.skipIf(FBX is None, "the Autodesk FBX SDK is required")
class TestTheModel(unittest.TestCase):
    """the read's model of the sdk against the sdk"""

    @classmethod
    def setUpClass(cls):
        cls.data = records()

    def test_get_r(self):
        """_fbx_get_r against FbxAMatrix.GetR on 400 blocks a row: plain,
        mirrored, sheared, sheared and mirrored, and from 1e-3 to exactly 0
        off gimbal"""
        rng = np.random.default_rng(11)

        def trs(rotate, scale):
            out = FBX.FbxAMatrix()
            out.SetTRS(FBX.FbxVector4(0, 0, 0), FBX.FbxVector4(*rotate), FBX.FbxVector4(*scale))
            return out

        def worst(made) -> float:
            blocks3 = np.array([matrix(item)[:3, :3] for item in made])
            want    = []
            for item in made:
                solved = FBX.FbxAMatrix()
                solved.SetR(item.GetR())
                want.append(matrix(solved)[:3, :3])
            return float(np.abs(H._fbx_get_r(blocks3) - np.array(want)).max())

        count = 400
        one   = rng.uniform(-170, 170, (count, 3))
        two   = rng.uniform(-170, 170, (count, 3))
        big   = rng.uniform(0.4,  2.5, (count, 3))
        wide  = rng.uniform(0.4,  2.5, (count, 3))
        none  = (0, 0, 0)
        rows = {
            "plain":            [trs(a, s) for a, s in zip(one, big)],
            "mirrored":         [trs(a, s * (-1, 1, 1)) for a, s in zip(one, big)],
            "sheared":          [trs(b, t) * trs(a, s) for a, s, b, t in zip(one, big, two, wide)],
            "sheared mirrored": [trs(b, t * (1, -1, 1)) * trs(a, s) for a, s, b, t in zip(one, big, two, wide)],
        }
        for off in (1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 0.0):
            near                          = one.copy()
            near[:, 1]                    = rng.choice([-90.0, 90.0], count) + off * rng.choice([-1.0, 1.0], count)
            rows[f"{off} off, one scale"] = [trs(a, (s[0], s[0], s[0])) for a, s in zip(near, big)]
            rows[f"{off} off, uneven"]    = [trs(a, s) for a, s in zip(near, big)]
            rows[f"{off} off, sheared"]   = [trs(none, t) * trs(a, s) for a, s, t in zip(near, big, wide)]
        for label, made in rows.items():
            with self.subTest(label):
                self.assertLessEqual(worst(made), 1e-14)

    def test_worlds(self):
        """_fbx_scene_worlds on the file's own scene against the sdk's
        EvaluateGlobalTransform, static and on keyed whole frames, a Z-up
        file among them, which reads no turn on a root; a scale of 0 is held
        as Maya holds it; _fbx_even gives Maya's verdict on 40 parents"""
        for name in ("s0_s123", "s0_sm123", "x_pv_s123", "ns_ksc1_nest_gs", "zup_x_pv_s123"):
            with self.subTest(name), H.open_fbx(fbx_path(name), FBX) as scene:
                tree, stack = _tree_and_take(scene)
                frames = np.arange(5, dtype=float)
                ticks  = H._fbx_ticks(frames / 24.0, H._fbx_ticks_per_second())
                held   = [np.repeat(values[None], len(frames), axis=0) for values in (tree.translate, tree.rotate, tree.scale)]
                H._fbx_sample_take(stack, _every_node(tree), ticks, tree, held)
                rows = list(range(1, len(tree.nodes)))
                rest = H._fbx_scene_worlds(tree, tree.translate[None], tree.rotate[None], tree.scale[None], rows, False, True)
                take = H._fbx_scene_worlds(tree, *held, rows, False, False)
                scene.SetCurrentAnimationStack(stack)
                for row in rows:
                    node = tree.nodes[row]
                    self.assertLessEqual(float(np.abs(rest[row][0] - matrix(node.EvaluateGlobalTransform())).max()), 1e-13)
                    for index, frame in enumerate(frames):
                        want = matrix(node.EvaluateGlobalTransform(frame_time(frame)))
                        self.assertLessEqual(float(np.abs(take[row][index] - want).max()), 1e-13)

        nodes = [
            spec("P", "group", None, s=(1, 0, 3)),
            spec("g", "group", "P", r=(20, 40, 60), inherit="Rrs"),
            spec("l", "locator", "g", t=(0, 1, 0)),
        ]
        with tempfile.TemporaryDirectory() as folder, H.open_fbx(written(folder, "zero", nodes), FBX) as scene:
            tree, _ = _tree_and_take(scene)
            rows   = list(range(1, len(tree.nodes)))
            worlds = H._fbx_scene_worlds(tree, tree.translate[None], tree.rotate[None], tree.scale[None], rows, False, True)
            self.assertTrue(all(np.isfinite(world).all() for world in worlds.values()))

        def euler(angles):
            return H._eulers_to_matrices(np.array(angles, dtype=float), 0)[:3, :3]

        parents = np.array([np.diag(s) @ euler(r) @ np.diag(t) @ euler(q) for s, r, t, q, _ in PARENTS])
        self.assertEqual(len(parents), 40)
        self.assertEqual(H._fbx_even(parents).tolist(), [not row[4] for row in PARENTS])

    def test_a_zero_is_positive(self):
        """Sig counts a zero as positive: an axis at 0 on a folded node takes
        the sign of the determinant of the others"""
        tree = types.SimpleNamespace(folds=np.array([False, True]), loose=frozenset())
        flip = H._fbx_flips(tree, np.array([[[1.0, 1.0, 1.0], [0.0, -1.0, 2.0]]]))
        self.assertEqual(flip[0, 1].tolist(), [-1.0, 1.0, -1.0])
        self.assertIsNone(H._fbx_flips(tree, np.array([[[-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]])))
        # a node the conversion leaves its keys keeps its file's scale
        tree.loose = frozenset((1,))
        self.assertIsNone(H._fbx_flips(tree, np.array([[[1.0, 1.0, 1.0], [0.0, -1.0, 2.0]]])))

    def test_a_left_handed_file_has_no_turn(self):
        """ConvertScene's turn is a right handed file's: a left handed one
        folds no root, a right handed one of the same axes does"""
        axes    = FBX.FbxAxisSystem
        manager = FBX.FbxManager.Create()
        self.addCleanup(manager.Destroy)
        scene = FBX.FbxScene.Create(manager, "scene")
        for hand, some in ((axes.ECoordSystem.eLeftHanded, False), (axes.ECoordSystem.eRightHanded, True)):
            with self.subTest(hand=hand):
                scene.GetGlobalSettings().SetAxisSystem(axes(axes.EUpVector.eZAxis, axes.EFrontVector.eParityEven, hand))
                self.assertEqual(H._fbx_converted_turn(scene) is not None, some)

    def test_left_keys(self):
        """_fbx_left_keys against the sdk's key reducer on 600 random curves,
        irregular times, each key mode: what it leaves and its last key, and
        the first on a level head"""
        modes   = FBX.FbxAnimCurveDef.EInterpolationType
        rng     = random.Random(11)
        manager = FBX.FbxManager.Create()
        scene   = FBX.FbxScene.Create(manager, "scene")
        self.addCleanup(manager.Destroy)

        def reduced(values, times, mode):
            curve = FBX.FbxAnimCurve.Create(scene, "curve")
            curve.KeyModifyBegin()
            for time, value in zip(times, values):
                key = curve.KeyAdd(frame_time(time))[0]
                curve.KeySetValue(key, float(value))
                curve.KeySetInterpolation(key, mode)
            curve.KeyModifyEnd()
            FBX.FbxAnimCurveFilterConstantKeyReducer().Apply(curve)
            place = {round(time * 16): index for index, time in enumerate(times)}
            kept  = [place[round(curve.KeyGetTime(i).GetSecondDouble() * 24.0 * 16)] for i in range(curve.KeyGetCount())]
            curve.Destroy()
            return kept

        for mode in (modes.eInterpolationLinear, modes.eInterpolationCubic, modes.eInterpolationConstant):
            for _ in range(200):
                values = _random_curve(rng)
                times  = np.cumsum([rng.choice([0.25, 0.5, 1, 1, 2, 3]) for _ in values]).tolist()
                kept   = reduced(values, times, mode)
                held   = np.array(values, dtype=np.float32).astype(float)[:, None]
                left, first, last = H._fbx_left_keys(held)
                self.assertEqual(bool(left[0]), bool(kept), values)
                if not kept:
                    continue
                self.assertEqual(int(last[0]), kept[-1], values)
                moved = np.flatnonzero(np.abs(held[:, 0] - held[0, 0]) > H._FBX_KEY_TOLERANCE)
                if len(moved) and moved[0] > 1 and not np.ptp(held[: moved[0], 0]):
                    self.assertEqual(int(first[0]), kept[0], values)

    def test_folded_angles(self):
        """_fbx_folded_angles against the sdk's own conversion, 100 random
        curves and 30 at the gimbal of each order in turn"""
        rng = random.Random(9)
        for _ in range(100):
            order  = rng.randrange(6)
            pre    = tuple(rng.randrange(-12, 13) * 7.5 for _ in range(3))
            frames = list(range(0, rng.randrange(4, 12)))
            step   = rng.choice([5.0, 15.0, 30.0, 60.0])
            values = np.zeros((len(frames), 3))
            now    = np.array([rng.uniform(-180, 180) for _ in range(3)])
            for frame in frames:
                values[frame] = now
                now           = now + np.array([rng.uniform(-step, step) for _ in range(3)])
            tree = _tree_of(order, pre)
            with self.subTest(order=order, pre=pre):
                got = H._fbx_folded_angles(tree, 1, values, None)
                self.assertLessEqual(float(np.abs(got - _sdk_angles(frames, values, order, pre)).max()), 1e-3)

        rng = random.Random(3)
        for order in range(6):
            first, middle, last = (int(axis) for axis in _axes_of(order))
            for _ in range(5):
                angles         = np.array([rng.randrange(-24, 25) * 7.5 for _ in range(3)])
                angles[middle] = rng.choice([90.0, -90.0])
                pre            = [0.0, 0.0, 0.0]
                pre[last]      = 30.0
                rotate         = angles.copy()
                rotate[last] -= 30.0
                values = np.array([rotate, rotate])
                want   = _sdk_angles([0, 1], values, order, tuple(pre))
                got    = H._fbx_folded_angles(_tree_of(order, tuple(pre)), 1, values, None)
                with self.subTest(order=order, angles=angles.tolist()):
                    self.assertLessEqual(float(np.abs((got - want + 180.0) % 360.0 - 180.0).max()), 1e-3)

        # through the gimbal of the order on a frame of the curve: the frame
        # shares what the free angle moved from the frame before, 2e-6 of 90
        # too, past the sdk's float32 and inside its 5e-6; 2**-16 and 2**-11
        # off 90 are no gimbal, but in xyz, where the band runs wider and the
        # angles stay one rotation, in the order scene C reads them
        for order in range(6):
            first, middle, last = (int(axis) for axis in _axes_of(order))
            for off in (0.0, 2e-6) + ((2.0**-16, 2.0**-11) if order else ()):
                values            = np.zeros((9, 3))
                values[:, first]  = 10.0 + 5.0 * np.arange(9)
                values[:, middle] = [70.0, 75.0, 80.0, 85.0, 90.0 - off, 95.0, 100.0, 105.0, 110.0]
                values[:, last]   = -20.0 + 7.5 * np.arange(9)
                pre               = [0.0, 0.0, 0.0]
                pre[last]         = 30.0
                frames            = list(range(9))
                want              = _sdk_angles(frames, values, order, tuple(pre))
                got               = H._fbx_folded_angles(_tree_of(order, tuple(pre)), 1, values, None)
                with self.subTest(order=order, off=off):
                    self.assertLessEqual(float(np.abs(got - want).max()), 1e-3)

    def test_what_the_conversion_leaves(self):
        """_fbx_reduced against the sdk's conversion with its key reduction
        on: a folded node's first and last key and the axes it empties"""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)

        # an axis level on the branch a(f) stays on, the other solution past
        # y = 90, where the principal solve carries it off
        past = spec(
            "g", "group", None, pre=(0, 0, 10), s=(1, 1, 1),
            keys={"r": (list(range(9)), [(30, 72 + 5 * frame, 0) for frame in range(9)])},
        )
        # two axes flip and the rotate turns a half turn about the third: S
        # moves where S Sig and a(f) stay level, so nothing is left
        flip = spec(
            "g", "group", None, pre=(35, 0, 0), s=(-1, 2, 3),
            keys={
                "s": ([0, 2, 3, 5], [(-1, 2, 3), (-1, 2, 3), (-1, -2, -3), (-1, -2, -3)]),
                "r": ([0, 2, 3, 5], [(0, 0, 0), (0, 0, 0), (180, 0, 0), (180, 0, 0)]),
            },
        )
        names = ("fl2_last", "fl2_head", "fl2_tail", "x_mx_zyx", "x_bs_zxy", "e_off_last")
        files = [(name, fbx_path(name)) for name in names]
        files += [("past", written(folder.name, "past", [past])), ("flip", written(folder.name, "flip", [flip]))]
        for name, path in files:
            with self.subTest(name), H.open_fbx(path, FBX) as scene:
                tree, stack = _tree_and_take(scene)
                rows  = [int(i) for i in np.flatnonzero(tree.folds)]
                every = list(range(1, len(tree.nodes)))
                keyed, axes, bounds, merged, keys, *_ = H._fbx_node_curves(tree, stack, every)
                tree.keyed = keyed
                rate       = 24.0 / H._fbx_ticks_per_second()
                own        = axes.any(axis=2)
                first      = np.where(own, bounds[..., 0], np.inf).min(axis=1)
                last       = np.where(own, bounds[..., 1], -np.inf).max(axis=1)
                frame      = np.arange(np.floor(first[keyed].min() * rate), np.ceil(last[keyed].max() * rate) + 1.0)
                ticks      = H._fbx_ticks(frame / 24.0, H._fbx_ticks_per_second())
                held       = [np.repeat(values[None], len(frame), axis=0) for values in (tree.translate, tree.rotate, tree.scale)]
                H._fbx_sample_take(stack, _every_node(tree), ticks, tree, held)
                flip = H._fbx_flips(tree, held[2])
                want = {}
                for i in [i for i in rows if keyed[i]]:
                    span    = (first[i] * rate, last[i] * rate)
                    curves  = [(axis, at * rate, values) for axis, at, values in keys.get(i, ())]
                    want[i] = H._fbx_reduced(tree, i, *held[1:], flip, frame, span, curves, merged[i])
                self.assertTrue(want)
                layer = stack.GetSrcObject(FBX.FbxCriteria.ObjectType(FBX.FbxAnimLayer.ClassId), 0)
                layer = H.typed(layer, FBX.FbxAnimLayer, FBX)
                _convert(scene, stack, True)
                for i, (start, stop, emptied, *_) in want.items():
                    got = _left_by_the_sdk(tree.nodes[i], layer)
                    with self.subTest(node=tree.nodes[i].GetName()):
                        if start is None:
                            self.assertIsNone(got[0])
                        else:
                            self.assertAlmostEqual(start, got[0], places=6)
                            self.assertAlmostEqual(stop, np.ceil(got[1] - 1e-6), places=6)
                        self.assertEqual(emptied.tolist(), got[2])


@unittest.skipIf(FBX is None, "the Autodesk FBX SDK is required")
class TestTheRead(unittest.TestCase):
    """what the rule costs and what it leaves alone"""

    @classmethod
    def setUpClass(cls):
        cls.data = records()

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name

    def test_a_bind_over_dropped_typed_nodes(self):
        """a bind pose listing a dropped parent over typed groups: a kept
        joint no pose lists takes the world the whole tree gives it, the
        typed group's static bake and shear in that tree"""
        kinds   = FBX.FbxTransform.EInheritType
        manager = FBX.FbxManager.Create()
        self.addCleanup(manager.Destroy)
        manager.SetIOSettings(FBX.FbxIOSettings.Create(manager, FBX.IOSROOT))
        scene = FBX.FbxScene.Create(manager, "scene")
        scene.GetGlobalSettings().SetAxisSystem(FBX.FbxAxisSystem.MayaYUp)

        def make(name, parent, joint, t, r, s, inherit):
            node = FBX.FbxNode.Create(scene, name)
            if joint:
                attribute = FBX.FbxSkeleton.Create(scene, name)
                attribute.SetSkeletonType(FBX.FbxSkeleton.EType.eLimbNode)
            else:
                attribute = FBX.FbxNull.Create(scene, name)
            node.SetNodeAttribute(attribute)
            node.LclTranslation.Set(FBX.FbxDouble3(*t))
            node.LclRotation.Set(FBX.FbxDouble3(*r))
            node.LclScaling.Set(FBX.FbxDouble3(*s))
            node.SetRotationActive(True)
            node.SetTransformationInheritType(inherit)
            (parent or scene.GetRootNode()).AddChild(node)
            return node

        P = make("P", None, False, (0, 1, 0),      (10, 30, -20), (1, 2, 3),      kinds.eInheritRSrs)
        g = make("g", P,    False, (1, 0.5, -0.5), (20, 40, 60),  (1, 1.5, 0.75), kinds.eInheritRrs)
        c = make("c", P,    False, (0, 1, 0),      (5, 0, 0),     (1, 1, 1),      kinds.eInheritRrSs)
        j = make("j", g,    True,  (0, 0, 1),      (10, 0, 0),    (1, 1, 1),      kinds.eInheritRrs)
        make("k", j, True, (0, 1, 0), (0, 20, 0), (1, 1, 1), kinds.eInheritRrs)
        make("e", c, True, (1, 0, 0), (0, 10, 0), (1, 1, 1), kinds.eInheritRSrs)

        pose = FBX.FbxPose.Create(scene, "bindPose")
        pose.SetIsBindPose(True)
        held = FBX.FbxMatrix()
        now  = P.EvaluateGlobalTransform()
        for row in range(4):
            for column in range(4):
                held.Set(row, column, now.Get(row, column) + (0.25 if (row, column) == (3, 1) else 0.0))
        pose.Add(P, held)
        scene.AddPose(pose)

        path     = os.path.join(self.folder, "binds.fbx")
        exporter = FBX.FbxExporter.Create(manager, "")
        exporter.Initialize(path, -1, manager.GetIOSettings())
        exporter.Export(scene)
        exporter.Destroy()

        whole = HierarchyData.load_fbx(path, pose="bind")
        kept  = HierarchyData.load_fbx(path, pose="bind", node_types=("joint",))
        self.assertTrue(np.any(whole["g"].shear))
        for name in ("j", "k", "e"):
            gap = np.abs(np.asarray(kept[name].world_matrix)[3, :3] - np.asarray(whole[name].world_matrix)[3, :3])
            self.assertLessEqual(float(gap.max()), 1e-12, name)

    def test_a_merged_channel_of_a_node_that_does_not_bake(self):
        """a node with a pre rotation keyed first before 0 off a whole frame:
        the conversion holds its rotate below the whole frame that key
        truncates to, but a node that does not bake reads a channel a later
        layer keys, muted or not, as the file's layers give it"""
        keys = {"r": ([-2.5, 0.5, 3.5], [(112.5, 30, 105), (37.5, -105, 60), (37.5, -120, 75)])}
        nodes = [
            spec("g", "group", None, t=(1, 2, 3), order="xzy", pre=(10, 0, 5), keys=keys),
            spec("l", "locator", "g", t=(0.5, 1, -2)),
        ]
        layer = [("additive", {"g": {"r": ([2, 6], [(5, 0, 0), (35, 0, 0)])}}, True)]
        for name, extra, held in (("held", {}, True), ("merged", {"layers": layer}, False)):
            path   = written(self.folder, name, nodes, span=(0, 8), **extra)
            clip   = ClipData.load_fbx(path, start_frame=-6, end_frame=0)
            rotate = np.asarray(clip.frames.rotate)[:, list(clip.name).index("g")]
            # frames -6 and -3 sit before the first key, -2 is the head
            self.assertTrue(np.allclose(rotate[0], rotate[3]), name)
            self.assertEqual(bool(np.allclose(rotate[3], rotate[4])), held, name)

    def test_a_tiny_scale_along_z_hands_its_shear_down(self):
        """a flat row along z, where the node's shear is not 0: what the
        row leaves it hands down through the inverse of that shear"""
        nodes = [
            spec("U", "none", None, t=(1, 2, 3), r=(20, 30, 40), s=(1, 2, 0.5)),
            spec("K", "locator", "U", t=(0, 0.5, 0.5), r=(0, 0, 30), s=(1, 1, 5e-13)),
            spec("A", "locator", "K", t=(0.3, 0.4, 0.2)),
        ]
        path = written(self.folder, "tiny", nodes)
        rig  = HierarchyData.load_fbx(path)
        with H.open_fbx(path, FBX) as scene:
            want = matrix(scene.FindNodeByName("A").EvaluateGlobalTransform())
        self.assertLessEqual(float(np.abs(np.asarray(rig["A"].world_matrix)[3, :3] - want[3, :3]).max()), 1e-9)

    def test_no_float32(self):
        """a static translate, a joint's pivots and a(f) keep the file's
        numbers, where Maya's import rounds them to single precision"""
        pivots = ((0.1, 0.2, 0.3), (0.3, 0.1, 0.2), (0.7, 0.1, 0.1), (0.1, 0.7, 0.3))
        nodes = [
            spec("P", "group", None, t=(0.1, 0.2, 0.3), r=(10, 30, -20), s=(2, 2, 2)),
            spec("g", "group", "P", t=(0.3, 0.7, 0.9), r=(20, 40, 60), inherit="RrSs"),
            spec("j", "joint", "P", t=(0.3, 0.7, 0.9), r=(11, 7, 13), pre=(3, 5, 7), s=(1.1, 1.3, 0.7), piv=pivots),
        ]
        rig = HierarchyData.load_fbx(written(self.folder, "exact", nodes))
        self.assertEqual(rig["P"].translate.tolist(), [0.1, 0.2, 0.3])
        self.assertEqual(rig["g"].translate.tolist(), [0.3, 0.7, 0.9])

        rp, sp, ro, so = np.array(pivots)
        scale = np.array([1.1, 1.3, 0.7])
        R     = H._eulers_to_matrices(np.array([11.0, 7.0, 13.0]), 0)[:3, :3]
        JO    = H._eulers_to_matrices(np.array([3.0, 5.0, 7.0]), 0)[:3, :3]
        want  = np.array([0.3, 0.7, 0.9]) + (ro + rp) + ((sp - rp) - sp * scale) @ R + so @ R @ JO
        self.assertLessEqual(float(np.abs(rig["j"].translate - want).max()), 1e-15)

        angles = np.array([[11.0, 7.0, 13.0], [41.0, 17.0, 3.0]])
        tree   = _tree_of(2, (3.0, 5.0, 7.0))
        got    = H._fbx_folded_angles(tree, 1, angles, None)
        block  = H._eulers_to_matrices(got, 2)[:, :3, :3]
        turned = H._eulers_to_matrices(angles, 2)[:, :3, :3] @ JO
        self.assertLessEqual(float(np.abs(block - turned).max()), 1e-14)

    def test_a_zero_and_a_nan(self):
        """a parent scaled exactly 0 on a frame and a NaN in a key: nothing
        raises, a node's channels on that frame are those Maya's floor of
        1e-12 gives, or its own block on the NaN, and the others bake as the
        same file with no zero and no NaN does"""
        turns = {"r": (list(range(7)), [(20 + 5 * frame, 40, 60) for frame in range(7)])}

        def nodes(scale):
            return [
                spec("P", "group", None, r=(10, 30, -20), s=(1, 2, 3), keys={"s": (list(range(7)), scale)}),
                spec("g", "group", "P", t=(1, 0, 0), r=(20, 40, 60), s=(1, 1.5, 0.75), inherit="RrSs", keys=turns),
                spec("h", "group", "P", t=(0, 1, 0), r=(20, 40, 60), s=(1, 1.5, 0.75), inherit="Rrs", keys=turns),
                spec("l", "locator", "g", t=(0, 1, 0)),
                spec("m", "locator", "h", t=(1, 0, 0)),
            ]

        plain    = [(1, 2, 3)] * 7
        zero     = list(plain)
        zero[3]  = (1, 0, 3)
        nan      = list(plain)
        nan[3]   = (float("nan"), 2, 3)
        floor    = list(plain)
        floor[3] = (1, 1e-12, 3)
        path     = written(self.folder, "plain", nodes(plain))
        want     = np.asarray(ClipData.load_fbx(path).frames.world_matrix)
        others   = [0, 1, 2, 4, 5, 6]

        clip = ClipData.load_fbx(written(self.folder, "zero", nodes(zero)))
        got  = np.asarray(clip.frames.world_matrix)
        self.assertLessEqual(float(np.abs(got[others] - want[others]).max()), 1e-12)
        self.assertTrue(np.isfinite(got[3]).all())
        for name in ("g", "h"):
            self.assertTrue(np.abs(clip.frames.shear[others, clip.index(name)]).max(axis=-1).all(), name)
        twin = ClipData.load_fbx(written(self.folder, "floor", nodes(floor)))
        for name in ("g", "h"):
            column = clip.index(name)
            for channel in ("scale", "shear", "rotate", "translate"):
                held = getattr(clip.frames, channel)[3, column] - getattr(twin.frames, channel)[3, column]
                self.assertLessEqual(float(np.abs(held).max()), 1e-9, (name, channel))

        clip = ClipData.load_fbx(written(self.folder, "nan", nodes(nan)))
        got  = np.asarray(clip.frames.world_matrix)
        self.assertLessEqual(float(np.abs(got[others] - want[others]).max()), 1e-12)
        self.assertTrue(np.isfinite(clip.frames.scale[3, clip.index("g")]).all())

    def test_nobody_pays(self):
        """no typed node and no folded one: no sdk read past the walk's and
        no world; the skin read none on a typed file; a typed file reads the
        curves of the nodes a bake needs alone"""
        from cgmath.geometry.skin_weights import load_fbx as skin_load_fbx

        names = ("_fbx_scene_worlds", "_fbx_node_curves", "_fbx_static_bakes")

        def counted():
            spies = {name: mock.patch.object(H, name, wraps=getattr(H, name)) for name in names}
            return spies, {name: spy.start() for name, spy in spies.items()}

        spies, calls = counted()
        try:
            for read in (HierarchyData, ClipData):
                read.load_fbx(fbx_path("x_nt_jp"))
            self.assertEqual(calls["_fbx_scene_worlds"].call_count, 0)
            self.assertEqual(calls["_fbx_node_curves"].call_count, 0)
            read = calls["_fbx_static_bakes"].call_count
            skin_load_fbx(fbx_path("x_sk_rrs"))
            self.assertEqual(calls["_fbx_scene_worlds"].call_count, 0)
            self.assertEqual(calls["_fbx_node_curves"].call_count,  0)
            self.assertEqual(calls["_fbx_static_bakes"].call_count, read)

            ClipData.load_fbx(fbx_path("x_sk_k"))
            rows = calls["_fbx_node_curves"].call_args[0][2]
            with H.open_fbx(fbx_path("x_sk_k"), FBX) as scene:
                tree, _ = _tree_and_take(scene)
                joint = [node.GetName() for node in tree.nodes].index("r0")
            self.assertEqual(calls["_fbx_node_curves"].call_count, 1)
            self.assertNotIn(joint, rows)
            with mock.patch.object(H, "_fbx_sample_take", wraps=H._fbx_sample_take) as sample:
                ClipData.load_fbx(fbx_path("x_sk_k"))
            self.assertEqual(sample.call_count, 2)
            self.assertNotIn(joint, [position for _, position in sample.call_args_list[1][0][1]])
        finally:
            for spy in spies.values():
                spy.stop()

    def test_old_save(self):
        """a save before 3.0.3 left a group eInheritRrSs: it reads where Maya
        reads it, 1.58 off the saved rig in the 3x3 and 3.17 in position,
        which is the sdk's world here, not where it was saved; a root stored
        eInheritRrs is typed like any node"""
        nodes = [
            spec("P", "group", None, r=(0, 0, 30), s=(1, 3, 0.5), inherit="RrSs"),
            spec("g", "group", "P", t=(1, 1, 0), r=(40, 10, -20), inherit="RrSs"),
            spec("l", "locator", "g", t=(0, 2, 1), inherit="RrSs"),
        ]
        path = written(self.folder, "old", nodes)
        back = HierarchyData.load_fbx(path)

        ids  = [f"00000000-0000-4000-8000-{index:012d}" for index in range(3)]
        saved = HierarchyData([
            TransformData("P", uuid=ids[0], rotate=(0, 0, 30), scale=(1, 3, 0.5)),
            TransformData("g", uuid=ids[1], parent_uuid=ids[0], translate=(1, 1, 0), rotate=(40, 10, -20)),
            TransformData("l", uuid=ids[2], parent_uuid=ids[1], translate=(0, 2, 1), node_type="locator"),
        ])
        axes = np.asarray(back["g"].world_matrix)[:3, :3] - np.asarray(saved["g"].world_matrix)[:3, :3]
        move = np.asarray(back["l"].world_matrix)[3, :3] - np.asarray(saved["l"].world_matrix)[3, :3]
        self.assertAlmostEqual(float(np.abs(axes).max()), 1.58, places=2)
        self.assertAlmostEqual(float(np.abs(move).max()), 3.17, places=2)
        self.assertTrue(np.any(back["g"].shear))

        with H.open_fbx(path, FBX) as scene:
            for node in (scene.FindNodeByName("g"), scene.FindNodeByName("l")):
                want = matrix(node.EvaluateGlobalTransform())
                self.assertLessEqual(float(np.abs(np.asarray(back[node.GetName()].world_matrix) - want).max()), 1e-12)

        rest = read_gaps(self.data, "tr_rrs_m", fbx_path("tr_rrs_m"))[1]
        self.assertLessEqual(max(max(value) for value in rest.values()), 1e-13)

    def test_far_keys(self):
        """a take keyed far from frame 0 pays for its keys: one row before
        the first key, one more _fbx_sample_take whatever that key's frame,
        and one more per distinct fraction of a first key before 0. A curve
        repeating before its keys keeps every row, as Maya's bake reads it"""
        take = "_fbx_sample_take"
        with mock.patch.object(H, take, wraps=getattr(H, take)) as sample:
            clip = ClipData.load_fbx(fbx_path("x_far"), start_frame=998, end_frame=1009)
        self.assertEqual(sample.call_count, 2)
        ticks = sample.call_args_list[1][0][2]
        self.assertEqual(len(ticks), 7)
        self.assertEqual(int(ticks[0]), 0)
        worlds = blocks(clip.frames.world_matrix)
        for column, node in enumerate(clip.name):
            key = f"x_far|{node}"
            if key in self.data:
                self.assertLessEqual(max(gap(worlds[:, column], self.data[key])), 1e-13, node)

        # one more call per distinct fraction of a first key before 0
        with mock.patch.object(H, take, wraps=getattr(H, take)) as sample:
            ClipData.load_fbx(fbx_path("x_hk_first"))
        self.assertEqual(sample.call_count, 3)

        with mock.patch.object(H, take, wraps=getattr(H, take)) as sample:
            clip = ClipData.load_fbx(fbx_path("x_cycle"), start_frame=0, end_frame=12)
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(len(sample.call_args_list[1][0][2]), 9)
        worlds = blocks(clip.frames.world_matrix)
        count  = 0
        for column, node in enumerate(clip.name):
            key = f"x_cycle|{node}"
            if key in self.data:
                count += 1
                self.assertLessEqual(max(gap(worlds[:, column], self.data[key])), 1e-13, node)
        self.assertGreaterEqual(count, 2)


def _tree_and_take(scene) -> tuple:
    """(the _FbxTree of an open scene, its first take)"""
    _, _, tree = H._walk_fbx(scene, H._fbx_units(scene, None))
    criteria = FBX.FbxCriteria.ObjectType(FBX.FbxAnimStack.ClassId)
    return tree, H.typed(scene.GetSrcObject(criteria, 0), FBX.FbxAnimStack, FBX)


def _every_node(tree) -> list:
    return [(node, position) for position, node in enumerate(tree.nodes)]


def _axes_of(order: int) -> tuple:
    from cgmath.transforms._numba._euler import MAYA_EA

    return tuple(MAYA_EA[order])


def _tree_of(order: int, pre: tuple):
    """what _fbx_folded_angles reads of a tree: node 1, keyed, under the scene root"""
    return types.SimpleNamespace(
        orders    = np.array([0, order]),
        pre       = np.array([(0.0, 0.0, 0.0), pre]),
        parent    = np.array([-1, 0]),
        keyed     = np.array([False, True]),
        axis_turn = None,
    )


def _random_curve(rng) -> list:
    """a curve of test_left_keys: noise, steps, ramps, a slow drift or a sine"""
    kind  = rng.choice(["noise", "steps", "ramps", "slow", "sin"])
    count = rng.randrange(1, 30)
    if kind == "noise":
        amount = rng.choice([0.001, 0.003, 0.005, 0.008, 0.012, 0.02])
        return [50.0 + rng.uniform(-amount, amount) for _ in range(count)]
    if kind == "steps":
        out, now = [], 10.0
        for _ in range(count):
            if rng.random() < 0.25:
                now += rng.choice([-1, 1]) * rng.uniform(0, 0.03)
            out.append(now)
        return out
    if kind == "ramps":
        out, now, slope = [], 3.0, 0.0
        for _ in range(count):
            if rng.random() < 0.2:
                slope = rng.choice([0.0, 0.0, 0.5, -0.5, 0.002, -0.002])
            now += slope
            out.append(now)
        return out
    if kind == "slow":
        step = rng.uniform(1e-4, 0.01)
        return [1.0 + step * index for index in range(count)]
    amount = rng.choice([0.01, 0.05, 0.2])
    speed  = rng.uniform(0.05, 0.6)
    return [20.0 + amount * float(np.sin(speed * index)) for index in range(count)]


def _convert(scene, stack, reduce_keys: bool) -> None:
    """the call Maya's import makes: every node's pre and post rotation to 0,
    its order kept, the curves re-keyed"""
    source, destination = FBX.FbxNode.EPivotSet.eSourcePivot, FBX.FbxNode.EPivotSet.eDestinationPivot
    active = FBX.FbxNode.EPivotState.ePivotActive

    def walk(node):
        node.SetPivotState(source, active)
        node.SetPivotState(destination, active)
        node.SetRotationOrder(destination, node.GetRotationOrder(source))
        node.SetPreRotation(destination, FBX.FbxVector4(0, 0, 0))
        node.SetPostRotation(destination, FBX.FbxVector4(0, 0, 0))
        for index in range(node.GetChildCount()):
            walk(node.GetChild(index))

    root = scene.GetRootNode()
    walk(root)
    root.ConvertPivotAnimationRecursive(stack, destination, 24.0, reduce_keys)


def _left_by_the_sdk(node, layer) -> tuple:
    """(first frame, last frame, [the rotate axes with no key]) a converted
    node's curves hold; (None, None, ...) where none holds a key"""
    frames  = []
    emptied = []
    for index, prop in enumerate((node.LclTranslation, node.LclRotation, node.LclScaling)):
        for axis in "XYZ":
            curve = prop.GetCurve(layer, axis, False)
            count = curve.KeyGetCount() if curve is not None else 0
            if count:
                frames += [curve.KeyGetTime(0).GetSecondDouble() * 24.0, curve.KeyGetTime(count - 1).GetSecondDouble() * 24.0]
            if index == 1:
                emptied.append(not count)
    if not frames:
        return None, None, emptied
    return round(min(frames), 6), round(max(frames), 6), emptied


def _sdk_angles(frames, values, order: int, pre: tuple) -> np.ndarray:
    """(F, 3) a(f) as the sdk's conversion re-keys it: one node, its rotate
    keyed on ``frames`` in ``order`` under the pre rotation ``pre``"""
    source  = FBX.FbxNode.EPivotSet.eSourcePivot
    names   = ("XYZ", "YZX", "ZXY", "XZY", "YXZ", "ZYX")
    manager = FBX.FbxManager.Create()
    scene   = FBX.FbxScene.Create(manager, "scene")
    scene.GetGlobalSettings().SetTimeMode(FBX.FbxTime.EMode.eFrames24)
    stack = FBX.FbxAnimStack.Create(scene, "take")
    layer = FBX.FbxAnimLayer.Create(scene, "layer")
    stack.AddMember(layer)
    scene.SetCurrentAnimationStack(stack)
    node = FBX.FbxNode.Create(scene, "g")
    node.SetNodeAttribute(FBX.FbxNull.Create(scene, "g"))
    node.SetRotationActive(True)
    node.SetRotationOrder(source, getattr(FBX.EFbxRotationOrder, "eEuler" + names[order]))
    node.SetPreRotation(source, FBX.FbxVector4(*pre))
    scene.GetRootNode().AddChild(node)
    for axis in range(3):
        add_curve(layer, node.LclRotation, axis, frames, values[:, axis])
    _convert(scene, stack, False)
    out = []
    for axis in "XYZ":
        curve = node.LclRotation.GetCurve(layer, axis, False)
        out.append([curve.KeyGetValue(key) for key in range(curve.KeyGetCount())])
    manager.Destroy()
    return np.array(out, dtype=float).T


if __name__ == "__main__":
    unittest.main()
