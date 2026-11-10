"""What Maya 2025 did, recorded once: the fixtures the fbx tests hold Maya to.

``test_assets/maya2025/<case>/`` holds, for each case, the files Maya
exported (``maya*.fbx``), the files our writer produced at recording time
and Maya imported (``ours*.fbx``), and ``readings.json`` + ``readings.npz``,
everything the tests asked Maya through ``cmds``: node types, parents,
attributes, world matrices frame by frame, deformed points, shader values.
``tools/make_maya_fbx_fixtures.py`` records them under mayapy; nothing here
needs Maya. ``manifest.json`` names the Maya that recorded them.
"""
import json
import os
import unittest

import numpy as np

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_assets", "maya2025")


class Readings:
    """one case's recording: ``array(key)`` a numpy array, ``value(key)`` a
    json value, ``path(name)`` a file the case holds"""

    def __init__(self, case: str):
        self.case   = case
        self.folder = os.path.join(FOLDER, case)
        with open(os.path.join(self.folder, "readings.json"), encoding="utf-8") as f:
            self.values = json.load(f)
        with np.load(os.path.join(self.folder, "readings.npz"), allow_pickle=False) as held:
            self.arrays = {key: held[key] for key in held.files}

    def array(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def value(self, key: str):
        return self.values[key]

    def path(self, name: str) -> str:
        return os.path.join(self.folder, name)

    def has(self, key: str) -> bool:
        return key in self.arrays or key in self.values


def recorded(case: str) -> Readings:
    """the case's readings; SkipTest when the case was never recorded"""
    folder = os.path.join(FOLDER, case)
    if not os.path.isfile(os.path.join(folder, "readings.json")):
        raise unittest.SkipTest(f"no Maya recording at {folder}: run tools/make_maya_fbx_fixtures.py under mayapy")
    return Readings(case)


def same_read(test: unittest.TestCase, got, want, label: str = "") -> None:
    """``got`` and ``want``, two ``load_fbx`` results, hold the same records
    field by field: the same None fields, the same counts, each record equal
    (uuids and names set aside, as ``==`` sets them). The guard that a fresh
    save still reads as the file Maya was given"""
    for field in want._fields:
        a, b = getattr(got, field), getattr(want, field)
        where = f"{label}{field}"
        if b is None or a is None:
            test.assertIs(a, b, where)
            continue
        if field == "clip":
            test.assertEqual(list(a.take), list(b.take), where)
            for x, y in zip(a, b):
                test.assertEqual(x.frame_count,          y.frame_count,          where)
                test.assertEqual((x.start_frame, x.fps), (y.start_frame, y.fps), where)
                test.assertEqual([n.name for n in x],    [n.name for n in y],    where)
                np.testing.assert_array_equal(np.asarray(x.frames.world_matrix), np.asarray(y.frames.world_matrix), err_msg=where)
                np.testing.assert_array_equal(np.asarray(x.frames.visibility), np.asarray(y.frames.visibility), err_msg=where)
            continue
        if field == "hierarchy":
            test.assertEqual([n.name for n in a], [n.name for n in b], where)
            test.assertEqual([n.parent_name for n in a], [n.parent_name for n in b], where)
            np.testing.assert_array_equal(np.asarray(a.world_matrix), np.asarray(b.world_matrix), err_msg=where)
            continue
        test.assertEqual(len(a), len(b), where)
        for i, (x, y) in enumerate(zip(a, b)):
            test.assertEqual(x, y, f"{where}[{i}]")
