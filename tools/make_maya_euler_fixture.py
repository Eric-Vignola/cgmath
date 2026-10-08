"""Record what Maya does with euler angles, for _tests/test_euler_maya.py.

usage (Windows, from the repository root):

    set MAYA_DISABLE_CER=1
    "C:/Program Files/Autodesk/Maya2025/bin/mayapy.exe" tools/make_maya_euler_fixture.py [out.npz]

The archive goes to _tests/test_assets/maya_euler.npz unless another path is
given. Every input comes from fixed seeds and the zip entries carry a fixed
date, so a second run with the same Maya writes the same bytes. Nothing here
imports cgmath: the inputs are built with numpy and Maya's own API, so the
code under test has no hand in its own reference.

Two records:

filterCurve -filter euler, run on the rx, ry and rz curves of a transform
    One curve per kind and rotate order, 48 keys (24 for "offset"):
    decomp8, decomp25, decomp60, decomp110 (a smooth random rotation at that
    many degrees per frame, decomposed by Maya), wrapflip (a smooth curve
    with random whole turns and flips added), gimbal (the middle axis around
    90) and offset (starting far outside -180..180). Then, per rotate order,
    25 two-key pairs where summing the axes and squaring them pick a
    different solution, and 5 two-key pairs with the middle axis at 90:
    tie0 an exact tie between the two solutions, tie1 to tie4 stepping off
    it until the other solution is 10, 20, 30 and 40 degrees shorter.

    filter_name   (C,)    "<order>_<kind>"
    filter_order  (C,)    Maya's rotate order, 0..5 = xyz yzx zxy xzy yxz zyx
    filter_start  (C+1,)  curve c's keys are filter_keys[start[c]:start[c+1]]
    filter_keys   (K, 3)  rx ry rz in degrees, as keyed
    filter_maya   (K, 3)  the same keys after the filter, as Maya reads them

xform -rotateOrder <order> -preserve on a joint with a joint orient
    40 rotations (the first 4 picked by hand, 36 random in -500..500) with
    joint orient (20, -30, 45), then 45 edge rotations with a random joint
    orient each: zero; one axis at +-180, +-90 or 720 and the others at zero;
    one axis at +-90, +-90 +- 1e-7 or +-180 and the others at 25, -40, 65;
    every axis at 180, every axis at -180, (180, -180, 90), and two
    whole-turn spins, (720, 30, -45) and (-720, 720, 15). Every rotation
    goes through all 30 pairs of rotate orders, on a new joint each time.

    reorder_rotate             (N, 3)     rotate before, degrees, as set
    reorder_from, reorder_to   (N,)       rotate order before and after
    reorder_joint_orient       (N, 3)     joint orient before, as Maya reads it
    reorder_maya               (N, 3)     rotate after, as Maya reads it
    reorder_joint_orient_after (N, 3)     joint orient after
    reorder_world_before       (N, 4, 4)  worldMatrix before
    reorder_world_change       (N, 4, 4)  worldMatrix after minus before;
                                          before + change is the after matrix
                                          exactly (checked on writing)

maya_version and maya_api_version say which Maya wrote it.

The keys and the edge rotations' joint orients are authored on a 1/256
degree grid, which compresses well; whatever Maya returns is stored as
returned.
"""
import math
import os
import random
import sys
import zipfile

import numpy as np

import maya.standalone

maya.standalone.initialize(name="python")

import maya.api.OpenMaya as om  # noqa: E402
import maya.cmds as cmds  # noqa: E402

HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(HERE, "..", "_tests", "test_assets", "maya_euler.npz")

ORDERS = ["xyz", "yzx", "zxy", "xzy", "yxz", "zyx"]

# the axes a rotate order applies, first to last; the middle one is the
# axis the other solution reflects rather than turns
APPLIED = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [1, 0, 2], [2, 1, 0]])

GRID = 2.0**8

# a zip entry's date is the time of writing unless given
DATE = (1980, 1, 1, 0, 0, 0)


def on_grid(values):
    return np.round(np.asarray(values, dtype=float) * GRID) / GRID


def other_solution(euler, order):
    """The second euler triple of the same rotation: the first and last
    applied axes turned by 180, the middle one reflected."""
    out = np.array(euler, dtype=float)
    first, middle, last = APPLIED[order]
    out[..., first] += 180.0
    out[..., last]  += 180.0
    out[..., middle] = 180.0 - out[..., middle]
    return out


def smooth_rotations(rng, frames, step):
    """Quaternions (x, y, z, w) of a rotation turning ``step`` degrees a
    frame about a slowly drifting axis."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    out = []
    for _ in range(frames):
        axis += rng.normal(size=3) * 0.15
        axis /= np.linalg.norm(axis)
        h = math.radians(step) / 2.0
        x1, y1, z1, w1 = (*(axis * math.sin(h)), math.cos(h))
        x2, y2, z2, w2 = q
        q = np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        )
        out.append(q)
    return out


def maya_euler(q, order):
    """Maya's own decomposition of a quaternion, in degrees."""
    euler = om.MQuaternion(*q).asEulerRotation().reorder(order)
    return [math.degrees(a) for a in (euler.x, euler.y, euler.z)]


def filter_cost_disagrees(keys, order):
    """True when summing the axes and squaring them would pick a different
    solution for the second of two keys."""
    previous, current = np.asarray(keys, dtype=float)
    candidates = np.stack((current, other_solution(current, order)))
    candidates += 360.0 * np.round((previous - candidates) / 360.0)
    step = np.abs(candidates - previous)
    l1, l2 = candidates[step.sum(-1).argmin()], candidates[(step**2).sum(-1).argmin()]
    return np.abs(l1 - l2).max() > 1e-3


def filter_curves():
    """name -> (rotate order, keys)"""
    curves = {}

    def add(kind, order, keys):
        curves[f"{ORDERS[order]}_{kind}"] = (order, on_grid(keys))

    for order in range(6):
        rng = np.random.default_rng(100 * order)
        mid = APPLIED[order][1]
        for speed in (8, 25, 60, 110):
            rotations = smooth_rotations(rng, 48, speed)
            add(f"decomp{speed}", order, [maya_euler(q, order) for q in rotations])

        keys = np.cumsum(rng.normal(size=(48, 3)) * 12.0, axis=0)
        keys += 360.0 * rng.integers(-2, 3, size=(48, 3))
        flip       = rng.random(48) < 0.3
        keys[flip] = other_solution(keys[flip], order)
        add("wrapflip", order, keys)

        keys         = np.cumsum(rng.normal(size=(48, 3)) * 15.0, axis=0)
        keys[:, mid] = 90.0 + rng.normal(size=48) * 3.0
        flip         = rng.random(48) < 0.3
        keys[flip]   = other_solution(keys[flip], order)
        add("gimbal", order, keys)

        keys = np.cumsum(rng.normal(size=(24, 3)) * 10.0, axis=0)
        add("offset", order, keys + [400.0, -530.0, 725.0])

    rng = np.random.default_rng(7)
    for order in range(6):
        found = 0
        while found < 25:
            previous = rng.uniform(-180.0, 180.0, 3)
            keys     = on_grid([previous, previous + rng.uniform(-170.0, 170.0, 3)])
            if filter_cost_disagrees(keys, order):
                add(f"pair{found}", order, keys)
                found += 1

        # tie0: the key and its other solution exactly as far from the one
        # before, 270 degrees summed either way. each k after it turns the
        # first axis 10 more and the last 5 more, which leaves the other
        # solution 10 * k degrees shorter: only tie0 is a tie
        first, middle, last = APPLIED[order]
        for k in range(5):
            keys            = np.zeros((2, 3))
            keys[1, first]  = 90.0 + 10.0 * k
            keys[1, last]   = -90.0 + 5.0 * k
            keys[1, middle] = 90.0
            add(f"tie{k}", order, keys)
    return curves


def record_filter():
    names, orders, start, keys, maya = [], [], [0], [], []
    for name, (order, values) in filter_curves().items():
        node = cmds.createNode("transform", name=name)
        cmds.setAttr(node + ".rotateOrder", order)
        for frame, row in enumerate(values):
            for attr, value in zip(("rx", "ry", "rz"), row):
                cmds.setKeyframe(node, attribute=attr, time=frame + 1, value=float(value))

        curves = [
            cmds.listConnections(f"{node}.{a}", type="animCurve")[0] for a in ("rx", "ry", "rz")
        ]
        cmds.filterCurve(*curves, filter="euler")
        filtered = np.array([cmds.keyframe(c, query=True, valueChange=True) for c in curves]).T
        assert filtered.shape == values.shape, name
        cmds.delete(curves + [node])

        names.append(name)
        orders.append(order)
        start.append(start[-1] + len(values))
        keys.append(values)
        maya.append(filtered)

    return {
        "filter_name":  np.array(names),
        "filter_order": np.array(orders, dtype=np.int8),
        "filter_start": np.array(start, dtype=np.int32),
        "filter_keys":  np.concatenate(keys),
        "filter_maya":  np.concatenate(maya),
    }


def reorder_rotations():
    """(rotate, joint orient) pairs"""
    # the 40 rotations of the first probe, with its joint orient
    random.seed(3)
    picked = [[400, 30, -200], [10, 95, 20], [0, 179, 0], [170, -60, 250]]
    out    = []
    for k in range(40):
        rotate = picked[k] if k < 4 else [random.uniform(-500, 500) for _ in range(3)]
        out.append((rotate, [20.0, -30.0, 45.0]))

    edges = [[0.0, 0.0, 0.0]]
    for axis in range(3):
        for value in (180.0, -180.0, 90.0, -90.0, 720.0):
            rotate       = [0.0, 0.0, 0.0]
            rotate[axis] = value
            edges.append(rotate)
    for axis in range(3):
        for value in (90.0, -90.0, 90.0 + 1e-7, 90.0 - 1e-7, -90.0 + 1e-7, -90.0 - 1e-7,
                      180.0, -180.0):
            rotate       = [25.0, -40.0, 65.0]
            rotate[axis] = value
            edges.append(rotate)
    edges += [
        [180.0, 180.0, 180.0],
        [-180.0, -180.0, -180.0],
        [180.0, -180.0, 90.0],
        [720.0, 30.0, -45.0],
        [-720.0, 720.0, 15.0],
    ]
    rng = np.random.default_rng(11)
    for rotate in edges:
        out.append((rotate, on_grid(rng.uniform(-90.0, 90.0, 3)).tolist()))
    return out


def record_reorder():
    rows = {k: [] for k in ("rotate", "from", "to", "jo", "maya", "jo_after", "before", "after")}
    for rotate, orient in reorder_rotations():
        for a in range(6):
            for b in range(6):
                if a == b:
                    continue
                node = cmds.createNode("joint")
                cmds.setAttr(node + ".rotateOrder", a)
                cmds.setAttr(node + ".rotate",      *rotate)
                cmds.setAttr(node + ".jointOrient", *orient)
                rows["jo"].append(cmds.getAttr(node + ".jointOrient")[0])
                rows["before"].append(cmds.getAttr(node + ".worldMatrix"))

                cmds.xform(node, rotateOrder=ORDERS[b], preserve=True)
                assert cmds.getAttr(node + ".rotateOrder") == b
                rows["maya"].append(cmds.getAttr(node + ".rotate")[0])
                rows["jo_after"].append(cmds.getAttr(node + ".jointOrient")[0])
                rows["after"].append(cmds.getAttr(node + ".worldMatrix"))
                rows["rotate"].append(rotate)
                rows["from"].append(a)
                rows["to"].append(b)
                cmds.delete(node)

    before = np.array(rows["before"]).reshape(-1, 4, 4)
    after  = np.array(rows["after"]).reshape(-1, 4, 4)
    change = after - before
    assert np.array_equal(before + change, after), "after - before does not give after back"

    return {
        "reorder_rotate": np.array(rows["rotate"], dtype=float),
        "reorder_from": np.array(rows["from"], dtype=np.int8),
        "reorder_to": np.array(rows["to"], dtype=np.int8),
        "reorder_joint_orient": np.array(rows["jo"]),
        "reorder_maya": np.array(rows["maya"]),
        "reorder_joint_orient_after": np.array(rows["jo_after"]),
        "reorder_world_before": before,
        "reorder_world_change": change,
    }


def save(path, arrays):
    """np.savez_compressed with a fixed date on every entry."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in arrays.items():
            info               = zipfile.ZipInfo(key + ".npy", date_time=DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(value), allow_pickle=False)


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else OUT)
    cmds.file(new=True, force=True)
    cmds.currentUnit(angle="degree", time="film")

    arrays = {
        "maya_version":     np.array(cmds.about(installedVersion=True)),
        "maya_api_version": np.array(cmds.about(apiVersion=True), dtype=np.int64),
    }
    arrays.update(record_filter())
    arrays.update(record_reorder())
    save(out, arrays)
    print(f"{out}: {os.path.getsize(out)} bytes")
    for key, value in arrays.items():
        print(f"    {key:28} {str(value.dtype):8} {value.shape}")


if __name__ == "__main__":
    main()
    maya.standalone.uninitialize()
