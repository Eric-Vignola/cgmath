"""Record where Maya's FBXImport puts every node, for cgmath's load_fbx tests.

usage (Windows, from the repository root, MAYA_DISABLE_CER=1 set):

    "C:/Program Files/Autodesk/Maya2025/bin/mayapy.exe" tools/make_fbx_maya_import_fixture.py [out.npz]
    "C:/Program Files/Autodesk/Maya2025/bin/mayapy.exe" tools/make_fbx_maya_import_fixture.py --limits <root> [out.npz]

The first writes the fbx files with the sdk, from the layouts below, in a
scratch folder, imports each one in Maya with FBXResetImport's settings, once
with animation on and once off, and keeps what load_fbx's tests read in
_tests/test_assets/fbx_maya_import.npz, unless another path is given. Every
layout is fixed and the zip entries carry a fixed date, so a second run with
the same Maya writes the same bytes. Nothing of it imports cgmath but the
writer's one module, _tests/_fbx_import_files.py, which needs the sdk alone:
the code under test has no hand in its own reference.

The second, run last, with <root> the folder holding the cgmath package,
reads every file through that cgmath and writes each file's position limit
into the archive: the largest position gap of its nodes, over the clip and
the rig, to the bit.

The archive holds, per file <f>, its name in "names", in the order below:

    <f>|layout    (B,)        uint8, the layout as json (_fbx_import_files.py)
    <f>|frames    (S,)        the frames Maya was read on: every whole frame of
                              the file's samples, the half frames too on the
                              files of test_half_frames (HALF)
    <f>|<n>       (S, 12)     the 4 x 3 of node n's world matrix on those frames,
                              for each leaf and each node stored eInheritRrSs
                              or eInheritRrs that is no joint
    <f>|s|<n>     (12,)       the same with animation off
    <f>|h|<n>     (S, 3)      Maya's shear, on the files of test_channels (SHEAR)
    <f>|p|<n>     (P, 3)      a mesh's object space points, on the x_me_* files

and "limits" (N,), each file's position limit, NaN until --limits has run;
maya_version and maya_api_version say which Maya wrote it. One more file,
x_cycle, for test_far_keys, is recorded the same way and is none of "names",
and so are the files of "edges", test_the_edges', each a rule the 180 cannot
see, read to Maya's own single precision: their values sit where the rule
needs them, not on the grid.

Every value of a test file sits on a grid single precision holds, between keys
too: Maya's import rounds to single precision and load_fbx does not, so a file
of free values measures Maya's rounding and not the rule. A keyed node with a
pre rotation holds a(f) on the grid only as a turn about one axis, its pre
about that axis, or a pre about the last axis of its order, whatever the three
angles. A joint's pivots hold it under quarter turns alone. Parent scales of
ev_* and fl_* sit on float32 values either side of a band, not on it.
"""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import zipfile

import numpy as np

import maya.standalone

maya.standalone.initialize(name="python")

import maya.api.OpenMaya as om  # noqa: E402
from maya import cmds, mel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "_tests", "test_assets", "fbx_maya_import.npz")
DATE = (1980, 1, 1, 0, 0, 0)

# whole frames and half frames, 0 to 6, which a read at twice the rate meets
SAMPLES = [i * 0.5 for i in range(13)]
HALF    = ("x_sp_s123", "x_sl_end14", "x_pv_s123", "x_pv_dc_G123_gs")
SHEAR   = ("s0_s123", "s0_s222", "x_kp_xyz_e")

FRAMES  = list(range(7))
TYPES   = ("RrSs", "RSrs", "Rrs")
ROT7    = [(20, 40, 60), (35, -10, 80), (-25, 50, 10), (60, 20, -40), (5, -70, 30), (-40, 25, -15), (15, 60, 45)]
SC7     = [(1, 2, 3), (1, 1, 1), (2, 1, 0.5), (1.5, 1.5, 1.5), (0.5, 2, 1), (1, 3, 2), (2, 2, 1)]
SCE7    = [(2, 2, 2), (1, 1, 1), (3, 3, 3), (0.5, 0.5, 0.5), (1.5, 1.5, 1.5), (2.5, 2.5, 2.5), (1, 1, 1)]
SCM7    = [(-1, 2, 3), (-2, 1, 1), (-0.5, 0.5, 2), (-1, 1, 1), (-3, 2, 1), (-1.5, 1, 0.5), (-2, 2, 2)]
NK_R    = [(10, 0, 20), (30, 10, 0), (-20, 40, 10), (0, -30, 50), (45, 15, -10), (20, 20, 20), (-10, 5, 35)]
TURN4   = [(10, 20, 30), (-40, 10, 5), (25, -60, 35), (70, 30, -10), (-15, 15, 80)]
SC1     = [(1, 2, 3), (1, 1, 1), (2, 1, 0.5), (1.5, 1.5, 1.5), (0.5, 2, 1)]
MESH    = [(0, 0, 0), (1, 0, 0.2), (0.3, 1, -0.1), (-0.4, 0.5, 0.6)]
EVERY   = "every"
FIVE    = list(range(5))


def node(name, kind, parent, t=(0, 0, 0), r=(0, 0, 0), s=(1, 1, 1), inherit=None, order="xyz", pre=None,
         post=None, piv=None, keys=None, ckeys=None, points=None):
    """a node dict: keys, {channel: (frames, values)}, all three components;
    ckeys, [(channel, axis, frames, values)]"""
    return dict(name=name, kind=kind, parent=parent, t=tuple(t), r=tuple(r), s=tuple(s),
                inherit=inherit or ("Rrs" if kind == "joint" else "RSrs"), order=order, pre=pre, post=post,
                piv=piv, keys=keys or {}, ckeys=ckeys or [], points=points)


def every(values):
    return (FRAMES, values)


def at(frames, values):
    return (list(frames), [values[f] for f in frames])


def strip(name):
    """a node's role: its name less the inherit type it hangs under"""
    for tag in TYPES:
        if name.endswith(tag):
            return name[:-len(tag)]
    return name


def std(P_s=(1, 2, 3), P_r=(0, 30, 0), P_keys=None, G=None, g_keys=EVERY, g_s=(1, 1.5, 0.75), nest=None):
    """the standard branch: an optional static group G at (0, 0.5, 0), a group
    P at (0, 1, 0), and under P a group g<T> per inherit type at (i, 0.5,
    -0.5), turned ROT7[0], over a locator, a turned group over a locator, a
    joint over a joint, a turned locator over joints, a turned and scaled
    locator over a group over joints; with ``nest``, typed nodes under g, and
    under them a joint. g_keys: EVERY keys ROT7 on every frame, None none"""
    lay = []
    if G is not None:
        lay.append(node("G", "group", None, t=(0, 0.5, 0), r=G[0], s=G[1]))
    lay.append(node("P", "group", "G" if G is not None else None, t=(0, 1, 0), r=P_r, s=P_s, keys=P_keys))
    for i, tag in enumerate(TYPES):
        keys = {} if g_keys is None else ({"r": every(ROT7)} if g_keys == EVERY else copy.deepcopy(g_keys))
        g    = "g" + tag
        lay.append(node(g, "group", "P", t=(i, 0.5, -0.5), r=ROT7[0], s=g_s, inherit=tag, keys=keys))
        lay += [node("l" + tag, "locator", g, t=(0.5, 1, -2)),
                node("c" + tag, "group", g, t=(1, 0, 0), r=(10, 20, 30)),
                node("d" + tag, "locator", "c" + tag, t=(0, 1, 0)),
                node("j" + tag, "joint", g, t=(0, 0, 1)),
                node("k" + tag, "joint", "j" + tag, t=(0, 1, 0)),
                node("e" + tag, "locator", g, t=(0, 0, 0.5), r=(0, 15, 0)),
                node("x" + tag, "joint", "e" + tag, t=(0, 0, 1)),
                node("y" + tag, "joint", "x" + tag, t=(0, 1, 0)),
                node("f" + tag, "locator", g, t=(0, 0.5, 0), r=(0, 0, 30), s=(1, 2, 0.5)),
                node("h" + tag, "group", "f" + tag, t=(0.5, 0, 0), r=(10, 20, 0)),
                node("i" + tag, "joint", "h" + tag, t=(0, 0, 1)),
                node("m" + tag, "joint", "i" + tag, t=(0, 1, 0))]
        if nest and tag != "RSrs":
            lay += [node("n1" + tag, "group", g, t=(0.5, 0, 0), r=(10, 0, 20), s=(1, 1.2, 0.8), inherit="RrSs"),
                    node("a1" + tag, "locator", "n1" + tag, t=(0, 1, 0)),
                    node("n2" + tag, "group", g, t=(0, 0.5, 0), r=(0, 25, 0), s=(0.7, 1, 1.3), inherit="Rrs"),
                    node("a2" + tag, "locator", "n2" + tag, t=(0, 1, 0)),
                    node("q" + tag, "group", g, t=(0, 0, 0.5), r=(15, 0, 0), s=(2, 2, 2)),
                    node("n3" + tag, "group", "q" + tag, t=(0.3, 0, 0), r=(0, 0, 40), s=(1, 0.5, 1.5), inherit="Rrs"),
                    node("a3" + tag, "locator", "n3" + tag, t=(0, 1, 0)),
                    node("j3" + tag, "joint", "n3" + tag, t=(0, 0, 1))]
    return lay


ORDERS = dict(G="zyx", P="zxy", gRrSs="yzx", gRSrs="zyx", gRrs="xzy", l="yxz", c="yxz", d="xzy", e="zxy", f="yzx",
              h="zyx", j="zyx", k="yzx", x="xzy", y="zxy", i="yxz", m="zyx")
JROT = dict(j=(15, -20, 30), k=(-10, 25, 5), x=(20, 10, -15), y=(5, 5, 40), i=(-30, 15, 10), m=(10, -5, 20),
            l=(25, -15, 40), d=(-20, 30, 10))
PREPOST = dict(G=((0, 10, 5), None), P=((10, 0, 15), (0, 20, 0)), gRrSs=((5, 10, 0), (0, 0, 15)),
               gRSrs=((-10, 5, 20), None), gRrs=((0, 15, 10), (10, 0, 0)), c=((0, 0, 25), None),
               f=(None, (15, 0, 0)), h=((5, 5, 5), (-5, 10, 0)), l=((20, 0, 0), None), j=((10, 0, 5), None),
               k=((0, 20, 0), None), x=(None, (0, 0, 10)), i=((5, 10, 15), (10, 0, 0)), y=((0, 0, -30), None))
PIVOTS = dict(P=((0.2, -0.3, 0.5), (0.1, 0.4, -0.2), (0.3, 0, -0.1), (-0.2, 0.1, 0.3)),
              G     = ((0.1, 0.2, 0), (0, -0.1, 0.3), (0, 0, 0.2), (0.1, 0, 0)),
              gRrSs = ((0.5, 0.2, 0), (-0.3, 0.1, 0.2), (0.1, 0.1, 0), (0, -0.2, 0.1)),
              gRSrs = ((0, 0.3, -0.4), (0.2, 0, 0.1), (0, 0.2, 0), (0.1, 0, -0.1)),
              gRrs  = ((-0.2, 0.1, 0.3), (0.3, -0.2, 0), (0.2, 0, 0.1), (0, 0.1, 0.2)),
              c     = ((0.1, 0, 0.2), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
              f     = ((0, 0, 0), (0.2, 0.3, -0.1), (0, 0, 0), (0.1, 0, 0)),
              h     = ((0.3, 0.1, 0), (0.1, 0, 0.2), (0.1, 0, 0), (0, 0.1, 0)),
              j     = ((0.1, 0.2, 0), (0, 0.1, 0), (0, 0, 0), (0, 0, 0)),
              i=((0, 0, 0), (0.2, 0, 0), (0, 0, 0.1), (0.1, 0, 0)))


def orders(lay):
    for n in lay:
        key = n["name"] if n["name"] in ORDERS else strip(n["name"])
        if key in ORDERS:
            n["order"] = ORDERS[key]
        if strip(n["name"]) in JROT and n["name"] not in ("G", "P"):
            n["r"] = JROT[strip(n["name"])]
    return lay


def prepost(lay):
    for n in lay:
        key = n["name"] if n["name"] in PREPOST else strip(n["name"])
        if key in PREPOST:
            n["pre"], n["post"] = PREPOST[key]
    return lay


def pivots(lay):
    for n in lay:
        key = n["name"] if n["name"] in PIVOTS else strip(n["name"])
        if key in PIVOTS:
            n["piv"] = PIVOTS[key]
    return lay


def jch(lay):
    """joints with their own rotate, scale and orient, and joints stored
    eInheritRrSs and eInheritRSrs under g"""
    own = dict(j=((15, -20, 30), (1.2, 0.8, 1.1), (5, 10, 15)), k=((-10, 25, 5), (0.9, 1.3, 1), None),
               x=((20, 10, -15), (1, 1, 1), (0, 30, 0)), y=((0, 0, 0), (1.1, 1.1, 0.7), None),
               i=((-30, 15, 10), (1, 1.5, 0.5), (10, 0, -20)), m=((10, -5, 20), (1, 1, 1), None))
    out = []
    for n in lay:
        role = strip(n["name"])
        if n["kind"] == "joint" and role in own:
            n["r"], n["s"], n["pre"] = own[role]
        out.append(n)
        if n["name"].startswith("g") and n["name"][1:] in TYPES:
            tag = n["name"][1:]
            out += [node("jb" + tag, "joint", n["name"], t=(0.2, 0, 0.8), r=(10, 20, 0), s=(1, 1.2, 0.9), inherit="RrSs"),
                    node("kb" + tag, "joint", "jb" + tag, t=(0, 1, 0), r=(0, 0, 15)),
                    node("jc" + tag, "joint", n["name"], t=(-0.2, 0, 0.8), r=(0, 15, -10), s=(1.1, 1, 1), inherit="RSrs"),
                    node("kc" + tag, "joint", "jc" + tag, t=(0, 0.5, 0.5))]
    return out


def tjoints(lay):
    """typed joints under P: eInheritRrSs chains, a keyed one, one with a joint orient"""
    return lay + [
        node("J1", "joint", "P", t=(0.5, 0, 0.5), r=(10, 20, 30), s=(1, 1.3, 0.8), inherit="RrSs"),
        node("J2", "joint", "J1", t=(0, 1, 0), r=(0, 0, 20), s=(1.2, 1, 1), inherit="RrSs"),
        node("J3", "joint", "J2", t=(0, 1, 0), inherit="Rrs"),
        node("J4", "joint", "J3", t=(0, 1, 0), r=(5, 0, 0), inherit="RSrs"),
        node("L1", "locator", "J1", t=(0, 0.5, 0.5)),
        node("H1", "group", "J1", t=(0.3, 0, 0), r=(0, 30, 0), s=(1, 2, 1)),
        node("J5", "joint", "H1", t=(0, 0, 1)),
        node("K1", "joint", "P", t=(-0.5, 0, 0.5), r=(0, 10, 0), s=(1, 1, 2), inherit="RSrs"),
        node("K2", "joint", "K1", t=(0, 1, 0), r=(20, 0, 0), inherit="RrSs"),
        node("K3", "joint", "K2", t=(0, 1, 0)),
        node("JR", "joint", "P", t=(0, 0, -0.5), r=(0, 0, 10), s=(1, 0.5, 1.5), inherit="RrSs", pre=(10, 20, 0)),
        node("JS", "joint", "JR", t=(0, 1, 0), inherit="Rrs"),
        node("JT", "joint", "P", t=(0, -0.5, 0), r=(0, 10, 0), inherit="Rrs"),
        node("JU", "joint", "JT", t=(0, 1, 0), r=(0, 0, 25), s=(1.5, 1, 1), inherit="RrSs"),
        node("J6", "joint", "P", t=(-0.5, 0.5, 0), r=ROT7[0], s=(1, 1.2, 0.9), inherit="RrSs", keys={"r": every(ROT7)}),
        node("J7", "joint", "J6", t=(0, 1, 0), inherit="Rrs"),
        node("J8", "joint", "J6", t=(0, 0, 1), r=(0, 20, 0), inherit="RrSs"),
    ]


def meshes(lay):
    m = [list(p) for p in MESH]
    return lay + [
        node("M1", "mesh", "P", t=(0.3, 0.2, 0.1), r=(10, 20, 30), s=(1, 1.2, 0.8), inherit="RrSs", points=m),
        node("M2", "mesh", "P", t=(-0.3, 0.2, 0.1), r=(10, 20, 30), s=(1, 1.2, 0.8), inherit="Rrs", points=m),
        node("M3", "mesh", "P", t=(0, 0.2, -0.3), r=(10, 20, 30), s=(1, 1.2, 0.8), points=m),
        node("M4", "mesh", "gRrSs", t=(0.2, 0.3, 0), r=(0, 30, 0), points=m),
        node("M5", "mesh", "gRrs", t=(0.2, 0.3, 0), r=(0, 30, 0), points=m),
        node("M6", "mesh", "P", t=(0.4, -0.2, 0), r=(5, 15, 25), s=(1.3, 0.7, 1), inherit="RrSs",
             pre=(10, 0, 20), piv=PIVOTS["h"], points=m),
        node("M8", "mesh", "fRrs", t=(0, 0.2, 0.2), r=(20, 0, 0), points=m),
        node("M9", "mesh", "gRrs", t=(0, -0.2, 0.3), s=(1, 0.5, 2), inherit="RrSs", points=m),
        node("M7", "mesh", "P", t=(0.1, 0.4, 0.2), r=ROT7[0], s=(1, 1.2, 0.8), inherit="RrSs",
             keys={"r": every(ROT7)}, points=m),
    ]


def grid(v, step=0.125):
    return tuple(round(float(x) / step) * step for x in v)


def ex(lay, joint_pivots=True):
    """every translate, pivot, point and translate key on 1/8"""
    for n in lay:
        n["t"] = grid(n["t"])
        if n["piv"] is not None:
            n["piv"] = tuple(grid(p) for p in n["piv"])
            if n["kind"] == "joint" and not joint_pivots:
                n["piv"] = None
        if n["points"] is not None:
            n["points"] = [list(grid(p)) for p in n["points"]]
        if "t" in n["keys"]:
            f, v = n["keys"]["t"]
            n["keys"]["t"] = (f, [grid(x) for x in v])
        n["ckeys"] = [(c, a, f, [round(float(x) * 8) / 8 for x in v]) for c, a, f, v in n["ckeys"]]
    return lay


def no_keyed_pre(lay):
    """a keyed node holds no pre rotation: re-keyed rotates are float32 of what no grid holds"""
    for n in lay:
        if n["keys"] or n["ckeys"]:
            n["pre"] = None
    return lay


def setn(lay, name, **over):
    for n in lay:
        if n["name"] == name:
            n.update(over)
    return lay


def typed_group(name):
    return name.startswith("g") and name[1:] in TYPES


FILES = {}


def add(name, lay, axes=("yup",), **extra):
    for ax in axes:
        full = name if ax == "yup" else ax + "_" + name
        fx   = dict(name=full, axes=ax, nodes=copy.deepcopy(lay), span=(0, 6))
        fx.update(extra)
        FILES[full] = fx


Y, Z, X = ("yup",), ("zup",), ("xup",)

# ---------------------------------------------------------------- stage 0: baked groups, frames 0 to 4
ROT5 = {"r": (FIVE, [list(v) for v in ROT7[:5]])}
s0   = dict(span=(0, 4))
add("s0_q38", std(P_s=(-1, 1, 1), G=((10, 0, 20), (1, 2, 3)), g_keys=ROT5), **s0)
add("s0_s2m1h_Gt", std(P_s=(2, -1, 0.5), G=((-15, 25, 5), (1, 2, 3)), g_keys=ROT5), **s0)
add("s0_s123", std(g_keys=ROT5), **s0)
add("s0_s222", std(P_s=(2, 2, 2), g_keys=ROT5), **s0)
add("s0_sm123", std(P_s=(-1, 2, 3), g_keys=ROT5), **s0)
add("s0_ksc1", std(P_keys={"s": (FIVE, [list(v) for v in SC1])}, g_keys=ROT5), **s0)
add("s0_dc_G123", std(P_s=(1, 1, 1), P_r=ROT7[0], P_keys=ROT5, G=((0, 0, 0), (1, 2, 3)), g_keys=ROT5), **s0)
add("s0_nest_s123", std(g_keys=ROT5, nest="static"), **s0)
add("s0_t4_G21h_gs", std(P_s=(1, 1, 1), P_r=TURN4[0], P_keys={"r": (FIVE, [list(v) for v in TURN4])},
                         G=((0, 0, 0), (2, 1, 0.5)), g_keys=None), **s0)
add("s0_zup_q38", std(P_s=(-1, 1, 1), G=((10, 0, 20), (1, 2, 3)), g_keys=ROT5), span=(0, 4))
add("s0_xup_q38", std(P_s=(-1, 1, 1), G=((10, 0, 20), (1, 2, 3)), g_keys=ROT5), span=(0, 4))
FILES["s0_zup_q38"]["axes"], FILES["s0_xup_q38"]["axes"] = "zup", "xup"
add("s0_s123_gs", std(g_keys=None), **s0)

# ---------------------------------------------------------------- orders, pre and post, mirrors
k_rt = {"r": every(ROT7)}
k_sc = {"s": every(SC7)}
s123 = lambda **k: std(P_s=(1, 2, 3), P_r=(10, 30, -20), **k)
smm  = lambda **k: std(P_s=(-1, 2, 3), P_r=(10, 30, -20), **k)
dc   = lambda **k: std(P_s=(1, 1, 1), P_r=ROT7[0], P_keys=k_rt, G=((10, 0, 20), (1, 2, 3)), **k)

add("ro_s123", orders(s123()))
add("ro_sm123_gs", orders(smm(g_keys=None)), Z)
add("ro_dc_G123_gs", orders(dc(g_keys=None)), X + Z)
add("ro_dc_G123_gk", orders(dc()))
add("pp_sm123_gs", prepost(smm(g_keys=None)), Y + Z)
add("x_pp_s123_gs", ex(no_keyed_pre(prepost(s123(g_keys=None)))))
add("x_pp_dc_G123_gs", ex(no_keyed_pre(prepost(dc(g_keys=None)))))

m = (-1, 1.5, 0.75)
add("gm_s123_gs", std(P_s=(1, 2, 3), g_s=m, g_keys=None))
add("gm_s222_Gt", std(P_s=(2, 2, 2), G=((10, 0, 20), (1, 1, 1)), g_s=m))
add("gm_ksceven", std(P_s=(2, 2, 2), P_keys={"s": every(SCE7)}, g_s=m))
add("gm_s222_G3", std(P_s=(2, 2, 2), G=((10, 0, 20), (3, 3, 3)), g_s=m), X)

add("ns_ksceven",      std(P_s=(2, 2, 2), P_keys={"s": every(SCE7)}))
add("ns_amk",          std(P_s=(-1, 2, 3), P_keys={"s": every(SCM7)}))
add("ns_amr_G123_gs",  std(P_s=(-1, 2, 3), P_r=ROT7[0], P_keys=k_rt, G=((0, 0, 0), (1, 2, 3)), g_keys=None))
add("ns_gmir",         std(P_s=(1, 2, 3), g_s=m))
add("ns_ksc1_nest_gs", std(P_s=(1, 2, 3), P_keys=k_sc, g_keys=None, nest="static"))
add("ns_ksc1_Gt123_gs", std(P_s=(1, 2, 3), P_keys=k_sc, G=((10, 0, 20), (1, 2, 3)), g_keys=None), X)

# ---------------------------------------------------------------- a bake's frames: spans, sparse keys
FR   = [float(f) for f in range(-4, 7)]
sp_g = {"r": at([0, 2, 6], ROT7)}
sp_t = lambda lay: [dict(n, ckeys=[("t", 1, [0, 4], [0.75, 0.25])] if n["name"] in ("gRrSs", "gRSrs", "gRrs") else n["ckeys"]) for n in lay]
add("sl_span26", std(P_s=(1, 2, 3), P_keys={"s": at([2, 4, 6], SC7)}, g_keys={"r": at([3, 5], ROT7)}), span=(2, 6))
add("x_sl_span04", std(P_s=(1, 2, 3), P_keys=k_sc, g_keys={"r": at([0, 2, 6], ROT7)}), span=(0, 4))
add("x_sl_end14", std(P_s=(1, 2, 3), P_keys=k_sc, g_keys={"r": at([1, 3], ROT7)}))
add("x_sl_neg", std(P_s=(1, 2, 3), P_keys={"s": ([-2, 2, 6], [SC7[0], SC7[2], SC7[4]])},
                    g_keys={"r": ([-2, 2], [ROT7[1], ROT7[3]])}), span=(-2, 6), samples=FR)
add("x_sln_e", std(P_s=(1, 2, 3), P_keys={"s": ([-2, 2, 6], [SC7[0], SC7[2], SC7[4]])},
                   g_keys={"r": ([2, 4], [ROT7[1], ROT7[3]])}), span=(-4, 6), samples=FR)
add("x_sp_s123",   ex(sp_t(std(P_s=(1, 2, 3), g_keys=sp_g))))
add("x_sp_ksc_gk", std(P_s=(1, 2, 3), P_keys={"s": at([0, 2, 4], SC7)}, g_keys={"r": at([2, 4, 6], ROT7)}))
add("x_sp_dc_gk",  std(P_s=(1, 1, 1), P_r=ROT7[0], P_keys={"r": at([0, 2, 4, 6], ROT7)}, G=((0, 0, 0), (1, 2, 3)), g_keys=sp_g))


# ---------------------------------------------------------------- a keyed node with a pre rotation
def turn(axis, angles):
    return [tuple(float(a) if i == axis else 0.0 for i in range(3)) for a in angles]


Z7 = turn(2, (10, 30, 50, 80, 120, 170, 190))
X7 = turn(0, (0, 20, 40, 60, 100, 150, 200))


def kp(order="xyz", P_s=(1, 2, 3), g_s=(1, 1.5, 0.75), post=None):
    lay = std(P_s=P_s, P_r=(0, 0, 0), P_keys={"r": every(Z7)}, g_keys={"r": every(X7)}, g_s=g_s)
    for n in lay:
        if n["name"] == "P":
            n["pre"], n["order"], n["post"] = (0, 0, 25), order, post
            n["r"] = Z7[0]
        if typed_group(n["name"]):
            n["pre"], n["order"], n["post"] = (35, 0, 0), order, post
            n["r"] = X7[0]
    return lay


add("x_kp_xyz", kp("xyz"))
add("x_kp_zxy", kp("zxy"), Y + Z)
add("x_kp_yzx", kp("yzx"), X)
add("x_kp_yzx_m",    kp("yzx", P_s=(-1, 2, 3)))
add("x_kp_zyx_gm",   kp("zyx", g_s=(-1, 1.5, 0.75)))
add("x_kp_zxy_post", kp("zxy", post=(0, 15, 0)))
add("x_kp_xyz_e",    kp("xyz", P_s=(2, 2, 2)))
add("x_kp_xyz_m", kp("xyz", P_s=(-1, 2, 3)), Z)
add("x_kp_zxy_m",  kp("zxy", P_s=(-1, 2, 3)))
add("x_kp_xyz_n",  kp("xyz", P_s=(-1, -2, 3)))
add("x_kp_zxy_n",  kp("zxy", P_s=(-1, -2, 3)))
add("x_kp_xyz_gn", kp("xyz", g_s=(1, -1.5, -0.75)))

# ---------------------------------------------------------------- pivots, joints, everything together
add("x_pv_s123", ex(pivots(s123())), Y + Z)
add("x_pv_sm123_gs",    ex(pivots(smm(g_keys=None))))
add("x_pv_dc_G123_gs",  ex(pivots(dc(g_keys=None))))
add("x_ropv_s123",      ex(pivots(orders(s123())), joint_pivots=False))
add("x_all_s123_gs",    ex(no_keyed_pre(pivots(prepost(orders(s123(g_keys=None))))), joint_pivots=False))
add("x_all_dc_G123_gs", ex(no_keyed_pre(pivots(prepost(orders(dc(g_keys=None))))), joint_pivots=False))
add("x_all_sm123_gs", ex(no_keyed_pre(pivots(prepost(orders(smm(g_keys=None))))), joint_pivots=False), Z)
add("x_jall_s123_gs", ex(no_keyed_pre(pivots(prepost(orders(jch(s123(g_keys=None)))))), joint_pivots=False))
add("x_tj_s123", ex(tjoints(s123())))
add("x_tj_sm123", ex(tjoints(smm())), X)
add("x_me_s123", ex(meshes(s123())))
add("x_me_dc_G123", ex(meshes(dc())), Z)
add("x_me_gs", ex(meshes(s123(g_keys=None))))

lay = std(P_s=(1, 2, 3))
lay += [node("JA", "joint", "P", t=(0.5, 0, 0.5), r=(10, 20, 30), s=(1, 1.3, 0.8)),
        node("HA", "group", "JA", t=(0, 1, 0), r=(0, 30, 0), s=(1, 2, 1), inherit="RrSs"),
        node("LA", "locator", "HA", t=(0, 0, 1)),
        node("HB", "group", "JA", t=(0, 0, 1), r=(30, 0, 0), s=(1, 2, 1), inherit="Rrs"),
        node("LB", "locator", "HB", t=(0, 0, 1)),
        node("JB", "joint", "P", t=(-0.5, 0, 0.5), r=(10, 20, 30), s=(1, 1.3, 0.8), inherit="RSrs"),
        node("HC", "group", "JB", t=(0, 1, 0), r=(0, 30, 0), s=(1, 2, 1), inherit="RrSs", keys={"r": every(ROT7)}),
        node("HD", "group", "JB", t=(0, 0, 1), r=(30, 0, 0), s=(1, 2, 1), inherit="Rrs", keys={"r": every(ROT7)})]
add("tj2_s123", lay)

# joint pivots: rotations of whole quarter turns, so tj sits on the grid
A8, B8 = (0.125, 0.25, 0.375), (-0.25, 0.125, 0.25)


def jp():
    ch  = dict(r=(90, 0, 0), pre=(0, 0, 90), post=(90, 0, 0), s=(1, 2, 0.5))
    lay = std(P_s=(1, 2, 3))
    rows = [("JP1", dict(piv=(A8, (0, 0, 0), (0, 0, 0), (0, 0, 0)), post=None, s=(1, 1, 1))),
            ("JP2", dict(piv=(A8, (0, 0, 0), (0, 0, 0), (0, 0, 0)), s=(1, 1, 1))),
            ("JP3", dict(piv=((0, 0, 0), B8, (0, 0, 0), (0, 0, 0)))),
            ("JP4", dict(piv=((0, 0, 0), (0, 0, 0), A8, (0, 0, 0)))),
            ("JP5", dict(piv=((0, 0, 0), (0, 0, 0), (0, 0, 0), A8))),
            ("JP6", dict(piv=(A8, B8, (0.25, -0.125, 0.125), (0.125, 0.125, -0.25)))),
            ("JP7", dict(piv=(A8, B8, (0.25, -0.125, 0.125), (0.125, 0.125, -0.25)), order="zxy", r=(0, 0, 90))),
            ("JP8", dict(piv=(A8, B8, (0, 0, 0), (0, 0, 0)), post=None,
                         keys={"r": every([(90.0 * (f % 4), 0, 0) for f in FRAMES])})),
            ("JP9", dict(piv=(A8, B8, (0.25, -0.125, 0.125), (0.125, 0.125, -0.25)), pre=None, post=None))]
    for i, (name, over) in enumerate(rows):
        d = dict(ch)
        d.update(over)
        lay.append(node(name, "joint", "P", t=(0.25 * i, 0, 0.5), **d))
        lay.append(node(name + "c", "joint", name, t=(0, 0.5, 0)))
    return lay


add("x_jp_s123", jp(), Y + Z)


# ---------------------------------------------------------------- roots typed like any node
def roots(inh, s=(1, 2, 3), keys=None, pre=None, post=None):
    return [node("R", "group", None, t=(1, 2, 3), r=(10, 20, 30), s=s, inherit=inh, keys=keys, pre=pre, post=post),
            node("A", "locator", "R", t=(0, 1, 0)),
            node("B", "group", "R", t=(1, 0, 0), r=(0, 30, 0), s=(1, 2, 1), inherit=inh),
            node("C", "locator", "B", t=(0, 0, 1)),
            node("J", "joint", "R", t=(0, 0, 1)),
            node("K", "joint", "J", t=(0, 1, 0))]


add("tr_rrs_m", roots("Rrs", s=(-1, 2, 3)), Y + X)
add("tr_rrss_m", roots("RrSs", s=(-1, 2, 3)))
add("tr_rrs_pp", roots("Rrs", pre=(10, 0, 5), post=(0, 15, 0)))
add("tr_rrs", roots("Rrs"), Z)
add("tr_rrss_pp", roots("RrSs", s=(-1, 2, 3), pre=(10, 0, 5), post=(0, 15, 0)), Z)
add("tr_rrs_k", roots("Rrs", keys=k_rt))
add("tr_rrss_k", roots("RrSs", keys=k_rt))


# ---------------------------------------------------------------- views, keys, rates, layers
def views(kind, inh, keys=None):
    return [node("G", "group", None, t=(0, 0.5, 0), r=(10, 0, 20), s=(1, 2, 3)),
            node("V", kind, "G", t=(1, 0.5, 0), r=(20, 40, 60), s=(1, 1.5, 0.75), inherit=inh, keys=keys),
            node("A", "locator", "V", t=(0.5, 1, -2)),
            node("B", "group", "V", t=(1, 0, 0), r=(10, 20, 30), inherit="RrSs"),
            node("J", "joint", "V", t=(0, 0, 1)),
            node("K", "joint", "J", t=(0, 1, 0))]


add("vw_camera_RrSs", views("camera", "RrSs"))
add("vw_light_Rrs",   views("light", "Rrs"))
add("vw_lod_k",       views("lod", "RrSs", keys=k_rt))
add("vw_none_RrSs",   views("none", "RrSs"))
add("vw_camera_k",    views("camera", "RrSs", keys=k_rt))
add("fr30_s123", std(P_s=(1, 2, 3)), fps=30)
add("fr60_s123", std(P_s=(1, 2, 3)), fps=60)
add("cu_ksc", std(P_s=(1, 2, 3), P_keys={"s": at([0, 2, 4, 6], SC7)}, g_keys=None), cubic=True)
add("la_add_sc", std(P_s=(1, 2, 3), g_keys=None), layers=[("additive", {"P": {"r": every([(0, 3 * f, 0) for f in FRAMES])}})])
add("la_over_s123", std(P_s=(1, 2, 3)), layers=[("override", {"gRrSs": {"r": every(NK_R)}})])

# ---------------------------------------------------------------- two negative axes
PV8 = ((0.25, 0.5, 0.75), (0.5, 0.25, 0.125), (0.125, 0.25, 0.5), (0.75, 0.5, 0.25))
PRE, POST = (10, 0, 5), (0, 15, 0)
add("x_nn_p", setn(std(P_s=(-1, -2, 3), P_r=(10, 30, -20), g_keys=None), "P", pre=PRE, post=POST), Y + Z)
add("x_nn_yz", setn(std(P_s=(1, -2, -3), P_r=(10, 30, -20), g_keys=None), "P", pre=PRE, post=POST))
add("x_nn_g", [dict(n, pre=PRE if n["name"].startswith("g") else n["pre"], post=POST if n["name"].startswith("g") else n["post"])
               for n in std(P_s=(1, 2, 3), P_r=(10, 30, -20), g_keys=None, g_s=(-1, -1.5, 0.75))], Y + X)
add("x_nn_pv", ex(setn(std(P_s=(1, 2, 3), P_r=(10, 30, -20), g_keys=None), "gRSrs", s=(-1, -1.5, 0.75), pre=PRE, piv=PV8)))


# ---------------------------------------------------------------- a far branch, a joint root, turned pivots
def far(order):
    return [node("T", "group", None, r=(10, 30, -20), s=(1, 1.5, 1)),
            node("Q", "group", "T", r=(10, 30, -20), keys={"r": every(ROT7)}),
            node("P", "group", "Q", t=(0.75, 2, -1.875), r=(-120, -10, 100), s=(1, 0.75, 0.75), order=order,
                 pre=(-40, 25, 20), inherit="RrSs"),
            node("g", "group", "P", t=(-0.5, -0.625, 0), r=(30, -50, -35), s=(1, 1, 1), inherit="RrSs",
                 keys={"s": ([0, 2, 4, 6], [(1, 1, 1), (1.5, 1, 1), (1.5, 1.5, 1), (1, 1.5, 1.5)])}),
            node("l", "locator", "g", t=(0.5, 1, -2)),
            node("j", "joint", "g", t=(0, 0.5, 1), r=(0, 30, 0)),
            node("k", "joint", "j", t=(0, 1, 0))]


for o in ("xzy", "zxy", "yxz"):
    add("x_br_" + o, far(o))


def jroot(pre=None, post=None):
    return [node("r", "joint", None, t=(1, 2, 3), r=(10, 20, 30), s=(1, 2, 3), pre=pre, post=post),
            node("a", "group", "r", t=(1, 1, 0), r=(45, 25, 95), s=(1, 2, 1)),
            node("c", "locator", "a", t=(-1.75, 1.875, 0.25), r=(-115, -55, -25), s=(1, 0.75, 3), inherit="RrSs"),
            node("d", "locator", "c", t=(1, 0, 0)),
            node("b", "group", "r", t=(1.5, 1.5, 0.75), r=(45, 25, 95), s=(0.25, 3, 1.25), inherit="RrSs"),
            node("e", "group", "b", t=(0.75, 1.25, 0.125), r=(5, 25, 80), s=(0.5, 2, 1.5), inherit="Rrs"),
            node("f", "locator", "e", t=(1, 0, 0))]


add("x_jr", jroot(), Z + X)
add("x_jrp", jroot(pre=(-60, 85, -90), post=(65, -65, 105)), Z)


def rootpv(keys=None, r=(10, 30, -20)):
    return [node("P", "group", None, t=(1, 2, 3), r=r, s=(1, 2, 3), inherit="Rrs", piv=PV8, keys=keys),
            node("c", "locator", "P", t=(0, 1, 0))]


add("x_rp_rrs", rootpv(), Z + X)
add("x_rpk_rrs", rootpv(r=X7[0], keys={"r": every(X7)}), Z)
add("x_rpk_rrs_z", rootpv(r=Z7[0], keys={"r": every(Z7)}), X)

# ---------------------------------------------------------------- the even test
MIRROR   = (-1, 1.5, 0.75)
even     = dict(P_r=(10, 30, -20), g_keys=None, g_s=MIRROR)
turned_z = dict(P_r=(0, 0, 45), g_keys=None, g_s=MIRROR)
add("ev_s13",   std(P_s=(1, 1, 1 + 2.0**-13), **even))
add("ev_s14",   std(P_s=(1, 1, 1 + 2.0**-14), **even))
add("ev_b13",   std(P_s=(4, 4, 4 + 2.0**-11), **even))
add("ev_b14",   std(P_s=(4, 4, 4 + 2.0**-12), **even))
add("ev_yz",    std(P_s=(1, 1 + 2.0**-14, 1 - 2.0**-14), **even))
add("ev_xz",    std(P_s=(1 - 2.0**-14, 1, 1 + 2.0**-14), **even))
add("ev_noise", [n for n in std(P_s=(1, 1 - 2.0**-24, 1 + 2.0**-23), **even) if not n["name"].endswith("Rrs")])
add("ev_h13",   std(P_s=(1, 1, 1), G=((0, 0, 0), (1, 1 + 2.0**-13, 1)), **turned_z))
add("ev_h14",   std(P_s=(1, 1, 1), G=((0, 0, 0), (1, 1 + 2.0**-14, 1)), **turned_z))
add("ev_h8",    std(P_s=(8, 8, 8), G=((0, 0, 0), (1, 1 + 2.0**-15, 1)), **turned_z))
add("ev_h8u",   std(P_s=(8, 8, 8), G=((0, 0, 0), (1, 1 + 2.0**-17, 1)), **turned_z))
add("ev_hx",    std(P_s=(1, 1, 1), P_r=(45, 0, 0), G=((0, 0, 0), (1, 1, 1 + 2.0**-13)), g_keys=None, g_s=MIRROR))


# ---------------------------------------------------------------- what counts as a scale curve
def fl(keys, static=(2, 2, 2), up=False):
    lay = std(P_s=(1, 1, 1) if up else static, P_r=(10, 30, -20), G=((0, 0, 0), static) if up else None, g_keys=None, g_s=MIRROR)
    return setn(lay, "G" if up else "P", keys=keys)


E17 = 2.0**-17


def same(v, *f):
    return (list(f), [(v, v, v)] * len(f))


add("fl_same",      fl({"s": same(2.0, 0, 5)}))
add("fl_one",       fl({"s": same(2.0, 3)}))
add("fl_other",     fl({"s": same(4.0, 0, 5)}))
add("fl_one_other", fl({"s": same(4.0, 3)}))
add("flq_same",     fl({"s": same(2.0, 0, 5)}, up=True))
add("flq_other",    fl({"s": same(4.0, 0, 5)}, up=True))
add("fl_near",      fl({"s": ([0, 2, 4], [(2, 2, 2), (2 + E17,) * 3, (2 - E17,) * 3])}))
add("fl_far",       fl({"s": ([0, 2, 4], [(2, 2, 2), (2 + E17,) * 3, (2 + 2 * E17,) * 3])}))

# ---------------------------------------------------------------- a first key before 0 off a whole frame
FN = [float(f) for f in range(-3, 9)]


def ramp(a, b):
    return ([a, b], [(0, 0, 0), (45, 0, 0)])


add("x_hk_first", std(P_s=(1, 2, 3), P_keys=k_sc, g_keys={"r": ramp(-1.5, 3)}), span=(-3, 6), samples=FN)
add("x_hk_fq",    std(P_s=(1, 2, 3), g_keys={"r": ramp(-1.25, 3.25)}),          span=(-3, 6), samples=FN)
add("x_hk_ft",    std(P_s=(1, 2, 3), g_keys={"r": ramp(-0.75, 3.75)}),          span=(-3, 6), samples=FN)
add("x_hk_fw",    std(P_s=(1, 2, 3), P_keys=k_sc, g_keys={"r": ramp(-2, 2.5)}), span=(-3, 6), samples=FN)
add("x_hk_pos",   std(P_s=(1, 2, 3), P_keys=k_sc, g_keys={"r": ramp(0.5, 5)}),  span=(-3, 6), samples=FN)

# ---------------------------------------------------------------- nothing typed, and still moved off 3.0.2
add("x_nt_mp", [node("r", "group", None, t=(1, 2, 3), r=(10, 30, -20)),
                node("a", "group", "r", t=(0, 1, 0), r=(20, 40, 60), s=(-1, 2, 3), pre=PRE, post=POST),
                node("b", "locator", "a", t=(0, 1, 0)),
                node("j", "joint", "a", t=(0, 0, 1)),
                node("k", "joint", "j", t=(0, 1, 0))])
add("x_nt_m", [node("r", "group", None, t=(1, 2, 3), r=(10, 30, -20), s=(-1, 2, 3)),
               node("b", "locator", "r", t=(0, 1, 0)),
               node("j", "joint", "r", t=(0, 0, 1)),
               node("k", "joint", "j", t=(0, 1, 0))], Z + X)
add("x_nt_jp", [node("r", "joint", None, t=(1, 2, 3), r=(90, 0, 0)),
                node("a", "joint", "r", t=(0, 1, 0), r=(90, 0, 0), pre=(0, 0, 90), post=(90, 0, 0), s=(1, 2, 0.5),
                     piv=(A8, B8, (0, 0, 0), (0, 0, 0))),
                node("b", "joint", "a", t=(0, 1, 0))])

# ---------------------------------------------------------------- the sdk's conversion: curves it empties
PX    = (35, 0, 0)
FM    = [float(f) for f in range(-4, 11)]
BELOW = dict(samples=FM, span=(0, 6))
WIDE  = dict(samples=FM, span=(-4, 10))


def g2(keys=None, P_keys=None, pre=PX, r=(0, 0, 0), order="xyz", typed=False):
    lay = std(P_s=(1, 2, 3), P_r=(0, 30, 0), P_keys=P_keys, g_keys=None)
    for n in lay:
        if typed_group(n["name"]):
            n["pre"], n["order"], n["r"], n["keys"] = pre, order, r, copy.deepcopy(keys or {})
            if typed and n["name"] == "gRSrs":
                n["keys"] = {}
    return lay


K_SC  = {"s": (list(range(0, 7)), [(1, 2 + 0.25 * f, 3 - 0.25 * f) for f in range(0, 7)])}
K_SCL = {"s": (list(range(-4, 11)), [(1 + 0.125 * (f + 4), 2, 3 - 0.125 * (f + 4)) for f in range(-4, 11)])}
T1    = {"t": ([3], [(2, 3, 4)])}
R1    = {"r": ([3], [(15, 0, 0)])}
S1    = {"s": ([3], [(2, 3, 4)])}
add("fl2_t", g2(T1), **BELOW)
add("fl2_r", g2(R1), **BELOW)
add("fl2_s", g2(S1), **BELOW)
add("fl2_tk", g2(T1, K_SC), **BELOW)
add("fl2_rk", g2(R1, K_SC), **BELOW)
add("fl2_sk", g2(S1, K_SC), **BELOW)
add("fl2_pair", g2({"r": ([3, 5], [(15, 0, 0), (15, 0, 0)])}, K_SC), **BELOW)
add("fl2_stat", g2({"t": ([0, 5], [(1, 0.5, -0.5), (1, 0.5, -0.5)])}, K_SC), **BELOW)
add("fl2_1ax", [dict(n, ckeys=[("r", 0, [3], [15.0])] if typed_group(n["name"]) else n["ckeys"]) for n in g2(None, K_SC)], **BELOW)
add("fl2_root_t", roots("Rrs", keys={"t": ([3], [(2, 3, 4)])}), Z + X, **BELOW)
add("fl2_root_r", roots("Rrs", keys={"r": ([3], [(10, 20, 45)])}), Z + X, **BELOW)
# beside a moving curve: no range from a level curve, one frame past the moving keys, the level value kept
RK = {"r": ([2, 4], [(10, 0, 0), (40, 0, 0)])}
add("fl2_first", g2(dict(RK, t=([-3], [(1, 0.5, -0.5)])), K_SCL), **WIDE)
add("fl2_last", g2(dict(RK, t=([8], [(1, 0.5, -0.5)])), K_SCL), **WIDE)
add("fl2_both", g2(dict(RK, t=([-3, 8], [(1, 0.5, -0.5), (1, 0.5, -0.5)])), K_SCL), **WIDE)
add("fl2_lv", g2(dict(RK, t=([-3], [(2, 1.5, 1)])), K_SCL), **WIDE)
add("fl2_lvs", g2(dict(RK, s=([-3, 8], [(2, 1.5, 0.75), (2, 1.5, 0.75)])), K_SCL), **WIDE)
add("fl2_tail", g2(dict(RK, t=([0, 4, 6, 8], [(1, 0.5, -0.5), (2, 0.5, -0.5), (2, 0.5, -0.5), (2, 0.5, -0.5)])), K_SCL), **WIDE)
add("fl2_head", g2(dict(RK, t=([-3, -2, 0, 4], [(1, 0.5, -0.5), (1, 0.5, -0.5), (1, 0.5, -0.5), (2, 0.5, -0.5)])), K_SCL), **WIDE)
# each side of the reducer's 0.009, two keys one frame apart
IN, PAST = 2.0**-7, 3 * 2.0**-8
for tag, d in (("in", IN), ("out", PAST)):
    add("fl2_bt_" + tag, g2({"t": ([3, 4], [(1, 0.5, -0.5), (1 + d, 0.5, -0.5)])}, K_SC, typed=True), **BELOW)
    add("fl2_bs_" + tag, g2({"s": ([3, 4], [(1, 1.5, 0.75), (1 + d, 1.5, 0.75)])}, K_SC, typed=True), **BELOW)
    add("fl2_br_" + tag, g2({"r": ([3, 4], [(10, 0, 0), (10 + d, 0, 0)])}, K_SC, typed=True),         **BELOW)
# keys either side of the last key's value, each within 0.009 of it: emptied, though their range is past it
add("fl2_anchor", g2({"t": ([1, 3, 5], [(1 - IN, 0.5, -0.5), (1 + IN, 0.5, -0.5), (1, 0.5, -0.5)])}, K_SC, typed=True), **BELOW)

# ---------------------------------------------------------------- the branch: the sum of squares
SEVEN = dict(samples=[float(f) for f in range(0, 7)])
BS = {"zxy": ([[-37.5, -37.5, -105.0], [37.5, -7.5, -82.5], [-112.5, 22.5, -142.5], [37.5, 52.5, -15.0], [-127.5, 82.5, 97.5]], (0, 30, 0)),
      "yzx": ([[-112.5, -52.5, 22.5], [-82.5, 15.0, -7.5], [-52.5, 150.0, 135.0], [-22.5, -135.0, -67.5]], (30, 0, 0)),
      "xzy": ([[67.5, 75.0, 97.5], [75.0, 90.0, 22.5], [120.0, 105.0, -37.5], [150.0, 120.0, -22.5], [120.0, 135.0, -22.5]], (0, 30, 0)),
      "yxz": ([[-120.0, 30.0, 30.0], [67.5, 150.0, 60.0], [75.0, -22.5, 90.0], [15.0, -60.0, 120.0], [-60.0, 82.5, 150.0]], (0, 0, 30))}
for o, (values, pre) in BS.items():
    add("x_bs_" + o, g2({"r": (list(range(len(values))), [tuple(v) for v in values])}, None, pre=pre, r=tuple(values[0]), order=o),
        span=(0, 4), **SEVEN)


# ---------------------------------------------------------------- an emptied axis: Maya's own solve, unbaked
def mx(order, pre, base, moving):
    keys = []
    for a in range(3):
        keys.append(("r", a, [0, 4, 8], [base[a] + (15.0 * i * (a + 1) if a in moving else 0.0) for i in range(3)]))
    return [node("P", "group", None, t=(1, 2, 3), r=tuple(base), order=order, pre=pre, ckeys=keys),
            node("l", "locator", "P", t=(0.5, 1, -2)),
            node("j", "joint", "P", t=(0, 0, 1)),
            node("k", "joint", "j", t=(0, 1, 0)),
            node("g", "group", "P", t=(1, 0.5, 0), r=(20, 0, 0), s=(1, 1.5, 0.75), inherit="RrSs",
                 keys={"t": ([2, 6], [(1, 0.5, 0), (2, 0.5, 0)])}),
            node("m", "locator", "g", t=(0.5, 1, -2))]


ELEVEN = dict(samples=[float(f) for f in range(0, 11)], span=(0, 8))
add("x_mx_zyx", mx("zyx", (-90, 0, 0), [10, 120, 30], (0, 2)), **ELEVEN)
under              = mx("zyx", (-90, 0, 0), [10, 120, 30], (0, 2))
under[0]["parent"] = "Q"
add("x_mx_q", [node("Q", "group", None, t=(0, 1, 0), r=(0, 30, 0), s=(1, 2, 3))] + under, **ELEVEN)
add("x_mx_yzx", mx("yzx", (-90, 0, 0), [10, 20, 120], (0, 1)), **ELEVEN)
add("x_mx_yxz", mx("yxz", (0, 0, -90), [120, 20, 30], (1, 2)), **ELEVEN)
add("x_mx_zxy", mx("zxy", (0, -90, 0), [120, 20, 30], (1, 2)), **ELEVEN)
add("x_mo_zyx", mx("zyx", (90, 0, 0), [10, 120, 30], (0, 2)), **ELEVEN)
add("x_mo_xyz", mx("xyz", (0, 0, -90), [10, 120, 30], (0, 2)), **ELEVEN)
add("x_mz_zyx", mx("zyx", None, [10, 120, 30], (0, 2)), Z, **ELEVEN)
add("x_mz_xyz", mx("xyz", None, [10, 120, 30], (0, 2)), X, **ELEVEN)


# ---------------------------------------------------------------- at the gimbal, far keys, a skin
def gb(order, pre, r):
    keys = {"r": ([0, 8], [tuple(r), tuple(r)]), "t": ([0, 8], [(1, 2, 3), (2, 2, 3)])}
    return [node("Q", "group", None, t=(0, 1, 0), r=(0, 30, 0), s=(1, 2, 3)),
            node("P", "group", "Q", t=(1, 2, 3), r=tuple(r), order=order, pre=pre, s=(1, 1.5, 0.75), keys=keys),
            node("g", "group", "P", t=(1, 0.5, 0), r=(20, 40, 60), s=(1, 1.5, 0.75), inherit="RrSs",
                 keys={"t": ([2, 6], [(1, 0.5, 0), (2, 0.5, 0)])}),
            node("m", "locator", "g", t=(0.5, 1, -2)),
            node("h", "group", "P", t=(0, 0.5, 0), r=(10, 0, 0), inherit="Rrs",
                 keys={"t": ([2, 6], [(0, 0.5, 0), (1, 0.5, 0)])}),
            node("n", "locator", "h", t=(0.5, 1, -2))]


add("x_gb_yzx", gb("yzx", (-30, 0, 0), (45, 30, 90)), **ELEVEN)
add("x_gb_yxz", gb("yxz", (0, 0, 30), (90, 30, 45)),  **ELEVEN)
add("x_gb_zyx", gb("zyx", (30, 0, 0), (45, -90, 30)), **ELEVEN)

FARK = list(range(1001, 1007))
add("x_far", std(P_s=(1, 2, 3), P_keys={"s": (FARK, [(1, 2 + 0.25 * i, 3) for i in range(6)])},
                 g_keys={"r": (FARK[1:3], [(0, 0, 0), (45, 0, 0)])}),
    samples=[float(f) for f in range(998, 1010)], span=(1001, 1006))


def sk(inh, P_s=(1, 2, 3), keys=None):
    lay = [node("r0", "joint", None, t=(0, 1, 0)),
           node("P", "group", None, t=(1, 2, 3), r=(10, 30, -20), s=P_s, keys=keys),
           node("g", "mesh", "P", t=(1, 0.5, 0), r=(20, 40, 60), s=(1, 1.5, 0.75), inherit=inh,
                points=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
           node("l", "locator", "g", t=(0.5, 1, -2)),
           node("j", "joint", "g", t=(0, 0, 1)),
           node("k", "joint", "j", t=(0, 1, 0))]
    lay[2]["skin"] = "r0"
    return lay


add("x_sk_rrs", sk("Rrs"), **SEVEN)
add("x_sk_rrss", sk("RrSs"), **SEVEN)
add("x_sk_e", sk("RrSs", P_s=(2, 2, 2)), **SEVEN)
add("x_sk_k", sk("RrSs", keys=k_rt), **SEVEN)

# no file of the 180: test_far_keys' curve repeating before its keys, frames 0 to 12
add("x_cycle", [dict(node("P", "group", None, t=(0, 1, 0), s=(1, 2, 3), keys={"r": ([4, 6, 8], turn(2, (0, 40, 20)))}), repeats=["r"]),
                node("g", "group", "P", t=(2, 0.5, -0.5), r=ROT7[0], s=(1, 1.5, 0.75), inherit="RrSs"),
                node("l", "locator", "g", t=(0.5, 1, -2)),
                node("j", "joint", "g", t=(0, 0, 1))],
    samples=[float(f) for f in range(0, 13)], span=(0, 12))
EXTRA = {"x_cycle": FILES.pop("x_cycle")}

# ---------------------------------------------------------------- no file of the 180: test_the_edges
EDGES = {}
TEN   = dict(samples=[float(f) for f in range(0, 11)], span=(0, 8))
LOW   = dict(samples=[float(f) for f in range(-8, 11)], span=(-6, 8))
KID   = dict(t=(1, 0.5, 0), r=(20, 40, 60), s=(1, 1.5, 0.75))
PN    = dict(t=(1, 2, 3), r=(10, 30, -20), s=(1, 2, 3))
GRN   = {"r": ([0, 8], [(20, 40, 60), (50, 40, 60)])}
R0    = {"r": ([0, 8], [(0, 0, 0), (40, 0, 30)])}
GROW  = {"s": (list(range(-6, 9)), [(1 + 0.125 * (i + 6), 2, 3 - 0.125 * (i + 6)) for i in range(-6, 9)])}
PIV   = ((0.25, 0.5, 0), (0.5, 0, 0.25), (0, 0.25, 0.25), (0.125, 0, 0.5))
PX5   = (10, 0, 5)
EVEN  = [dict(t=(0, 0, 0), s=(1, 2, 4)), dict(s=(1, 0.5, 0.25))]


def tail(par):
    return [node("l", "locator", par, t=(0.5, 1, -2)), node("j", "joint", par, t=(0, 0.5, 1), r=(0, 30, 0)),
            node("k", "joint", "j", t=(0, 1, 0)), node("m", "group", "k", t=(1, 0, 0))]


def edge(name, lay, axes="yup", **extra):
    fx = dict(name=name, axes=axes, nodes=copy.deepcopy(lay), span=(0, 6))
    fx.update(extra)
    EDGES[name] = fx


def under(P_keys=None, P_s=(1, 0.5, 0.25), **g):
    """Q over P over a typed g, P's world even at the static pose"""
    return [node("Q", "group", None, **EVEN[0]), node("P", "group", "Q", s=P_s, keys=P_keys), node("g", "group", "P", **dict(KID, **g))] + tail("g")


# the even test reads the static pose: even there and uneven on the frames,
# g stays as stored
edge("e_even_key", under(R0, inherit="RrSs", keys=GRN), **TEN)
edge("e_even_full", under({"r": ([0, 8], [(0, 0, 0), (360, 0, 0)])}, inherit="RrSs", keys=GRN), **TEN)
edge("e_even_flat", under({"r": R0["r"], "s": ([0, 8], [(1, 1, 1), (1, 1, 1)])}, inherit="RrSs", keys=GRN), **TEN)
edge("e_even_mirror", under({"r": ([3, 4, 6], turn(2, (0, 30, 60)))}, inherit="RrSs", s=(-1, 1.5, 0.75)))
# uneven at the static pose, even on the frames: it bakes
edge("e_rest", [node("P", "group", None, t=(0, 1, 0), r=(0, 30, 0), s=(1, 2, 3), keys={"s": ([0, 6], [(2, 2, 2), (2, 2, 2)])}),
                node("g", "group", "P", t=(1, 0.5, -0.5), r=(20, 40, 60), s=(-1, 1.5, 0.75), inherit="RrSs"),
                node("l", "locator", "g", t=(0.5, 1, -2))])
# a scale curve moving above the parent bakes
edge("e_climb", [node("Q", "group", None, t=(0, 1, 0), keys={"s": ([0, 2, 4, 6], [(1, 1, 1), (2, 2, 2), (1.5, 1.5, 1.5), (3, 3, 3)])}),
                 node("P", "group", "Q", t=(0, 1, 0), r=(0, 30, 0)),
                 node("g", "group", "P", t=(1, 0.5, -0.5), r=(20, 40, 60), s=(-1, 1.5, 0.75), inherit="RrSs"),
                 node("l", "locator", "g", t=(0.5, 1, -2))])
# a dropped typed node of identity channels still divides its parent's scale out
edge("e_still", [node("P", "group", None, t=(0, 1, 0), r=(0, 30, 0), s=(1, 2, 3)), node("g", "group", "P", inherit="Rrs"),
                 node("j", "joint", "g", t=(1, 0.5, 0)), node("k", "joint", "j", t=(0, 1, 0))])
# two live layers: the conversion reads the first layer's translate keys
edge("e_layers", [node("P", "group", None, t=(0, 1, 0), r=(0, 30, 0), s=(1, 2, 3)),
                  node("g", "group", "P", t=(1, 0, 0), pre=(0, 0, 90), inherit="RrSs",
                       keys={"t": ([0, 2, 4, 6], [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)])}),
                  node("l", "locator", "g", t=(0.5, 1, -2)),
                  node("Q", "group", None, t=(5, 0, 0), keys={"r": ([0, 6], [(0, 0, 0), (0, 0, 0)])}),
                  node("n", "locator", "Q", t=(0, 1, 0))],
     layers=[("additive", {"Q": {"r": ([0, 3, 6], [(0, 0, 0), (0, 45, 0), (0, 90, 0)])}})])


def skin(inh, keys=None, P_s=(1, 2, 3), P_pre=None, parent="P", **over):
    """a mesh g skinned to a joint, stored ``inh`` under P"""
    lay = [node("r0", "joint", None, t=(0, 1, 0)), node("P", "group", None, t=(1, 2, 3), r=(10, 30, -20), s=P_s, pre=P_pre)]
    lay.append(node("g", "mesh", parent, inherit=inh, keys=keys, points=MESH, **dict(KID, **over)))
    lay[-1]["skin"] = "r0"
    return lay + [node("l", "locator", "g", t=(0.5, 1, -2)), node("j", "joint", "g", t=(0, 0, 1))]


# a skinned mesh that bakes: its static bake read again, a keyed one baked
edge("e_skin_mirror", skin("RrSs", s=(-1, 1.5, 0.75)), **TEN)
edge("e_skin_mirror_pre", skin("Rrs", pre=PX5, s=(-1, 1.5, 0.75)), **TEN)
edge("e_skin_pre", skin("Rrs", pre=PX5), **TEN)
edge("e_skin_keyed", skin("RrSs", keys={"r": ([0, 4, 8], [(20, 40, 60), (35, 40, 60), (50, 55, 60)])}), **TEN)
edge("e_skin_root", skin("Rrs", parent=None), axes="zup", **TEN)
edge("e_skin_pivots", skin("RrSs", pre=(30, 20, 5), piv=PIV), **TEN)
edge("e_skin_all", skin("Rrs", order="yxz", pre=PX5, post=(0, 45, 0), piv=PIV), **TEN)
edge("e_skin_flip", skin("Rrs", P_s=(-1, 2, 3), P_pre=PX5), **TEN)
edge("e_skin_empty", [dict(n, skin_empty=True) if n["name"] == "g" else n for n in skin("RrSs")], **TEN)

# a bake runs to the later of 0 and its last key: on whole frames, and from
# a first key before 0 off a whole frame
lowg = dict(KID, pre=PX5, inherit="RrSs")
edge("e_before_zero", [node("P", "group", None, keys=GROW, **PN), node("g", "group", "P", keys={"r": ([-5, -3], [(20, 40, 60), (50, 45, 60)])}, **lowg)] + tail("g"), **LOW)
edge("e_before_all", [node("P", "group", None, keys={"r": ([-5, -3], [(10, 30, -20), (40, 30, -20)])}, **PN),
                      node("g", "group", "P", inherit="RrSs", keys={"r": ([-3.5, -2.75], [(20, 40, 60), (50, 45, 60)])}, **KID)] + tail("g"), **LOW)
# a last key off a whole frame on a folded node: the bake runs to the frame after it
edge("e_off_last", [node("P", "group", None, **PN), node("g", "group", "P", keys={"r": ([0.5, 2.5, 4.5], [(20, 40, 60), (40, 50, 65), (70, 20, 30)])}, **lowg)] + tail("g"),
     samples=[float(f) for f in range(-4, 9)], span=(0, 8))
# keys all before the whole frame the first truncates to: the conversion
# leaves them as they are, with no pre rotation, read in xyz below
edge("e_loose_bake", [node("P", "group", None, keys=GROW, **PN), node("g", "group", "P", keys={"r": ([-1.5], [(20, 40, 60)])}, **lowg)] + tail("g"), **LOW)
edge("e_loose_two", [node("P", "group", None, keys=GROW, **PN), node("g", "group", "P", keys={"r": ([-1.75, -1.25], [(20, 40, 60), (50, 45, 60)])}, **lowg)] + tail("g"), **LOW)
edge("e_loose_order", [node("P", "group", None, keys=GROW, **PN), node("g", "group", "P", order="zxy", pre=PX5, keys={"r": ([-1.5], [(20, 40, 60)])}, **KID),
                       node("h", "group", "g", inherit="RrSs", t=(0, 1, 0), r=(5, 10, 15), s=(1, 2, 1))] + tail("h"), **LOW)
edge("e_loose_mirror", [node("P", "group", None, **PN), node("g", "group", "P", pre=PX5, inherit="Rrs", keys={"r": ([-2.5], [(20, 40, 60)])}, **dict(KID, s=(-1, 1.5, 0.75)))] + tail("g"), **LOW)
edge("e_loose_root", [node("g", "group", None, keys={"r": ([-1.5], [(20, 40, 60)])}, **KID),
                      node("h", "group", "g", inherit="RrSs", t=(0, 1, 0), s=(1, 2, 1))] + tail("h"), axes="zup", **LOW)
edge("e_loose_scale", [node("P", "group", None, keys=GROW, **PN), node("g", "group", "P", inherit="Rrs", pre=PX5, keys={"s": ([-0.5], [(1, 2, 0.5)])}, **KID)] + tail("g"), **LOW)
edge("e_loose_free", [node("P", "group", None, keys=GROW, **PN), node("g", "group", "P", pre=PX5, keys={"r": ([-1.5], [(20, 40, 60)])}, **dict(KID, s=(-1, 1.5, 0.75)))] + tail("g"), **LOW)

# a node with no attribute under a joint is a joint in Maya's import
JN = dict(t=(1, 2, 3), r=(10, 30, -20), s=(1, 2, 3))


def bare(J_keys=None, **g):
    """g, with no attribute, under a joint J"""
    return [node("J", "joint", None, keys=J_keys, **JN), node("g", "none", "J", **dict(KID, **g)), node("l", "locator", "g", t=(0.5, 1, -2)),
            node("c", "group", "g", inherit="RrSs", r=(0, 30, 0), s=(1, 2, 1)), node("e", "joint", "c", t=(0, 1, 0))]


edge("e_bare", bare(inherit="RrSs"), **TEN)
edge("e_bare_rrs", bare(inherit="Rrs"), **TEN)
edge("e_bare_keyed", bare(inherit="RrSs", keys=GRN), **TEN)
edge("e_bare_rrs_keyed", bare(inherit="Rrs", keys=GRN), **TEN)
edge("e_bare_pre", bare(pre=PX5, post=(0, 15, 0)), **TEN)
edge("e_bare_moving", bare({"s": (list(range(9)), [(1 + 0.125 * f, 2, 3 - 0.125 * f) for f in range(9)])}, inherit="Rrs"), **TEN)
edge("e_bare_child", bare(inherit="RrSs") + [node("h", "none", "g", inherit="RrSs", t=(0, 1, 0), s=(2, 1, 1)), node("q", "locator", "h", t=(0.5, 1, -2))], **TEN)

# a joint root with pivots in a file ConvertScene turns: its turn is a pre
jroot_piv = [node("J", "joint", None, t=(1, 2, 3), r=(10, 30, -20), piv=PIV), node("u", "group", "J", r=(0, 30, 0), s=(1, 2, 0.5)),
             node("g", "group", "u", inherit="Rrs", **KID)] + tail("g")
edge("e_joint_pivot_z", jroot_piv, axes="zup", **TEN)
edge("e_joint_pivot_x", jroot_piv, axes="xup", **TEN)


def through(order, frames=9):
    """a keyed node through the gimbal of its order on frame 4, a pre about its last axis, a typed node under it"""
    first, middle, other = dict(xyz=(0, 1, 2), yzx=(1, 2, 0), zxy=(2, 0, 1), xzy=(0, 2, 1), yxz=(1, 0, 2), zyx=(2, 1, 0))[order]
    pre, keys = [0.0, 0.0, 0.0], []
    pre[other] = 30.0
    for f in range(frames):
        a = [0.0, 0.0, 0.0]
        a[first], a[middle], a[other] = 10.0 + 5 * f, 70.0 + 5.0 * f, -20.0 + 7.5 * f
        keys.append(tuple(a))
    return [node("Q", "group", None, **PN), node("g", "group", "Q", inherit="Rrs", t=(1, 0.5, 0), order=order, pre=tuple(pre), keys={"r": (list(range(frames)), keys)}),
            node("l", "locator", "g", t=(0.5, 1, -2))]


# through the gimbal of its order mid-curve: the solution nearest the frame before
edge("e_gimbal_zxy",  through("zxy"),    **TEN)
edge("e_gimbal_yzx",  through("yzx"),    **TEN)
edge("e_gimbal_xzy",  through("xzy"),    **TEN)
edge("e_gimbal_yxz",  through("yxz"),    **TEN)
edge("e_gimbal_zyx",  through("zyx"),    **TEN)
edge("e_gimbal_last", through("zxy", 5), **TEN)


# the conversion re-keys a folded node from the whole frame its first key
# truncates to: below it, it holds that frame's rotate and scale, and its
# angles take their branch from there, on the whole frames and on the steps
# of a bake from a first key before 0 off a whole frame
UF   = [(20, 40, 60), (40, 50, 65), (70, 20, 30)]
LOWS = dict(samples=[float(f) for f in range(-5, 11)], span=(0, 8))
C05  = {"r": ([-1.5, 0.5, 4.5], [(112.5, 30, 105), (37.5, -105, 60), (37.5, -120, 75)])}
GK   = {"r": ([-2.5, 3], [(20, 40, 60), (50, 45, 60)])}
edge("e_head_rotate", [node("g", "group", None, t=(1, 2, 3), s=(1, 1.5, 0.75), r=(20, 40, 60), pre=PX5,
                            keys={"r": ([-2.5, 0.5, 3.5], UF)}), node("l", "locator", "g", t=(0.5, 1, -2))], **LOWS)
edge("e_head_scale", [node("g", "group", None, t=(1, 2, 3), s=(1, 1.5, 0.75), r=(20, 40, 60), pre=PX5,
                           keys={"s": ([-2.5, 0.5, 3.5], [(1, 1.5, 0.75), (2, 1.5, 0.75), (2, 1, 0.75)])}),
                      node("l", "locator", "g", t=(0.5, 1, -2))], **LOWS)
for name, inh in (("e_head_bake", "Rrs"), ("e_head_bake_rrss", "RrSs")):
    edge(name, [node("P", "group", None, keys=GROW, **PN),
                node("g", "group", "P", inherit=inh, order="xzy", pre=(82.5, -45, 75), keys=C05, **KID)] + tail("g"), **LOW)
for name, order, pre in (("e_head_step", "xzy", (82.5, -45, 75)), ("e_head_step_xyz", "xyz", PX5), ("e_head_step_zyx", "zyx", (30, 60, -45))):
    edge(name, [node("Q", "group", None, keys=GROW, **PN), node("P", "group", "Q", t=(0, 1, 0), order=order, pre=pre, keys=C05),
                node("g", "group", "P", inherit="Rrs", keys=GK, **KID)] + tail("g"), **LOW)
# the steps read the parent held below its head: its turn alone, so the
# conversion's keys between whole frames are its own
edge("e_head_step_rrss", [node("Q", "group", None, keys=GROW, **PN),
                          node("P", "group", "Q", t=(0, 1, 0), pre=(30, 0, 0), keys={"r": ([-1.5, 0.5, 4.5], [(10, 0, 0), (40, 0, 0), (100, 0, 0)])}),
                          node("g", "group", "P", inherit="RrSs", keys=GK, **KID)] + tail("g"), **LOW)

# a node keyed in a layer past the first, muted or not, bakes on every whole
# frame of the take's span, from the earlier of 0 and its start, past the key
# reduction, its values those of the layers that are not muted
FR   = list(range(-4, 13))
PM   = {"s": (FR, [(1 + 0.0625 * (f + 4), 2, 3 - 0.0625 * (f + 4)) for f in FR])}
R26  = {"r": ([2, 6], [(5, 0, 0), (35, 0, 0)])}
MUTE = dict(samples=[float(f) for f in FR])


def layered(span, muted=True, inherit="RrSs", base=None, pre=None):
    """a typed g under a parent that moves, keyed by a second layer"""
    return dict(nodes=[node("P", "group", None, keys=PM, **PN), node("g", "group", "P", inherit=inherit, pre=pre, keys=base, **KID)] + tail("g"),
                span=span, layers=[("additive", {"g": R26}, True) if muted else ("additive", {"g": R26})])


for name, lay in (("e_mute", layered((0, 8))), ("e_mute_span", layered((-2, 8))), ("e_mute_late", layered((2, 8), inherit="Rrs")),
                  ("e_mute_base", layered((0, 8), base={"t": ([1, 3], [(1, 0.5, 0), (2, 0.5, 0)])})), ("e_mute_pre", layered((0, 8), pre=PX5)),
                  ("e_live", layered((0, 4), muted=False))):
    edge(name, lay.pop("nodes"), **dict(MUTE, **lay))

# the joint Maya makes of a node with no attribute under a joint keeps no
# pivot; the pre and post rotation of one whose keys the conversion leaves
# as its joint orient and rotate axis; an inactive one's pre rotation as its
# joint orient
NK = dict(t=(1, 0.5, 0), r=(20, 40, 60), s=(1, 1.5, 0.75))
JS = dict(t=(0, 1, 0), r=(10, 20, 30), s=(1, 2, 0.5))
NJ = dict(samples=[float(f) for f in range(-5, 11)], span=(-5, 8))


def below(par):
    return [node("g2", "group", par, inherit="RrSs", t=(0.5, 0.25, 0), r=(10, 0, 20), s=(1, 2, 1)), node("l2", "locator", par, t=(0.5, 1, -0.5)),
            node("k2", "joint", par, t=(0, 1, 0), r=(0, 30, 0)), node("m2", "group", "k2", t=(1, 0, 0))]


for name, piv in (("e_bare_pivots_rp", ((0, 1, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))), ("e_bare_pivots_so", ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 1, 0)))):
    edge(name, [node("J", "joint", None, t=(0, 1, 0), r=(10, 20, 30)), node("N", "none", "J", t=(1, 0.5, 0), r=(20, 40, 60), piv=piv),
                node("l", "locator", "N", t=(0.5, 1, -0.5)), node("k", "joint", "N", t=(0, 1, 0))], **TEN)
edge("e_bare_loose_pre", [node("J", "joint", None, **JS), node("N", "none", "J", keys={"r": ([-2.5], [(50, 55, 30)])}, **dict(NK, pre=PX5))] + below("N"), **NJ)
edge("e_bare_loose_pp", [node("J", "joint", None, **JS),
                         node("N", "none", "J", keys={"s": ([-2.5], [(1.5, 1, 1.25)])}, **dict(NK, pre=PX5, post=(0, 15, 0)))] + below("N"), **NJ)
edge("e_bare_inactive", [node("J", "joint", None, **JS), dict(node("N", "none", "J", **dict(NK, pre=PX5, post=(0, 15, 0))), active=False)] + below("N"), **NJ)

# a skinned mesh under a camera root of a file ConvertScene turns, where the
# sdk places it, the camera's rotate axis left out
CAM = dict(t=(1, 2, 3), r=(10, 30, -20))


def skinned(name, parent, inherit="RSrs", **over):
    return dict(node(name, "mesh", parent, inherit=inherit, points=MESH, **dict(KID, **over)), skin="r0")


edge("e_camera_skin", [node("r0", "joint", None, t=(0, 1, 0)), node("C", "camera", None, s=(1, 2, 3), **CAM), skinned("g", "C")] + tail("g"),
     axes="zup", **TEN)
edge("e_camera_skin_yup", [node("r0", "joint", None, t=(0, 1, 0)), node("C", "camera", None, s=(1, 2, 3), **CAM), skinned("g", "C")] + tail("g"),
     **TEN)
edge("e_camera_skin_typed", [node("r0", "joint", None, t=(0, 1, 0)), node("C", "camera", None, **CAM), skinned("g", "C", "RrSs", pre=PX5)] + tail("g"),
     axes="zup", **TEN)
edge("e_camera_skin_deep", [node("r0", "joint", None, t=(0, 1, 0)), node("C", "camera", None, **CAM),
                            node("h", "group", "C", t=(0, 1, 0), r=(0, 0, 30)), skinned("g", "h")] + tail("g"), axes="zup", **TEN)
edge("e_camera_skin_keyed", [node("r0", "joint", None, t=(0, 1, 0)),
                             node("C", "camera", None, keys={"r": ([0, 4, 8], [(10, 30, -20), (25, 30, -20), (25, 45, -20)])}, **CAM),
                             skinned("g", "C")] + tail("g"), axes="xup", **TEN)

# an emptied axis takes Maya's solve on the frame nearest 0, not the first
# key's: keyed from -4, where the moving axes sit far from frame 0's


def first_far(order, pre, base, far, mid):
    keys = [("r", a, [-4, 0, 4, 8], [far[a], base[a], base[a] + 15.0 * (a + 1), base[a] + 30.0 * (a + 1)] if a != mid else [base[a]] * 4)
            for a in range(3)]
    return [node("P", "group", None, t=(1, 2, 3), r=tuple(base), order=order, pre=pre, ckeys=keys), node("l", "locator", "P", t=(0.5, 1, -2)),
            node("j", "joint", "P", t=(0, 0, 1)), node("k", "joint", "j", t=(0, 1, 0)),
            node("g", "group", "P", t=(1, 0.5, 0), r=(20, 0, 0), s=(1, 1.5, 0.75), inherit="RrSs", keys={"t": ([2, 6], [(1, 0.5, 0), (2, 0.5, 0)])}),
            node("m", "locator", "g", t=(0.5, 1, -2))]


EARLY = dict(samples=[float(f) for f in range(-4, 9)], span=(-4, 8))
edge("e_mixed_first", first_far("zxy", (0, -90, 0), [120, 20, 30], [120, 151.5, 30], 0), **EARLY)
edge("e_mixed_first_zyx", first_far("zyx", (-90, 0, 0), [10, 120, 30], [113.7, 120, 35.2], 1), **EARLY)

# a later layer merges each channel it keys over the take's span, after the
# conversion has run on the first layer: a channel it does not key keeps
# its own frames, the conversion's on a folded node (a head, a lone key it
# leaves loose); a curve the conversion empties holds its first key's value
# under the merge; a merged channel holds before and after the span, a
# folded node's rotate on the branch the conversion gave it
SW     = list(range(-8, 17))
PW     = {"s": (SW, [(1 + 0.0625 * (f + 8), 2, 3 - 0.0625 * (f + 8)) for f in SW])}
SPREAD = {"r": ([-4, 2, 12], [(20, 40, 60), (35, 50, 60), (80, 20, 30)])}
HEAD   = {"r": ([-2.5, 0.5, 3.5], [(112.5, 30, 105), (37.5, -105, 60), (37.5, -120, 75)])}
S26    = {"s": ([2, 6], [(0, 0, 0), (0.5, 0, 0)])}
T26    = {"t": ([2, 6], [(0.5, 0, 0), (1.5, 1, 0)])}
MW     = dict(samples=[float(f) for f in range(-6, 15)])


def merging(g, keys, span=(0, 8)):
    """g under a parent that moves, a muted second layer keying ``keys`` on it"""
    return dict(nodes=[node("P", "group", None, keys=PW, **PN), g] + tail("g"), span=span, layers=[("additive", {"g": keys}, True)])


for name, lay in (("e_merge_base", merging(node("g", "group", "P", inherit="RrSs", keys={"t": ([-3, 0, 11], [(1, 0.5, 0), (2, 0.5, 0), (1, 1.5, 0)])},
                                                **KID), R26, (2, 8))),
                  ("e_merge_other", merging(node("g", "group", "P", inherit="RrSs", keys=SPREAD, **KID), S26)),
                  ("e_merge_loose", merging(node("g", "group", "P", pre=PX5, keys={"r": ([-2.5], [(50, 55, 30)])}, **KID), R26)),
                  ("e_merge_head", merging(node("g", "group", "P", inherit="Rrs", order="xzy", pre=PX5, keys=HEAD, **KID), T26)),
                  ("e_merge_head_rotate", merging(node("g", "group", "P", inherit="Rrs", order="xzy", pre=PX5, keys=HEAD, **KID), R26)),
                  ("e_merge_off", merging(node("g", "group", "P", inherit="RrSs", **KID), R26, (-2.5, 7.5))),
                  ("e_merge_held", merging(node("g", "group", "P", inherit="RrSs", keys=dict(SPREAD, t=([-3, 11], [(1, 0.5, 0), (1, 1.5, 0)])), **KID),
                                           R26, (2, 8))),
                  ("e_merge_held_turn", merging(node("g", "group", "P", inherit="Rrs", order="xzy", pre=PX5,
                                                     keys={"r": (HEAD["r"][0] + [12], HEAD["r"][1] + [(60, -100, 80)]),
                                                           "t": ([-4, 11], [(1, 0.5, 0), (1, 1.5, 0)])}, **KID), R26, (1, 8)))):
    edge(name, lay.pop("nodes"), **dict(MW, **lay))
edge("e_merge_emptied", [node("P", "group", None, **PN),
                         node("g", "group", "P", inherit="RrSs", pre=PX5, keys={"t": ([0, 4, 8], [(1, 0.5, 0), (1 + 2**-8, 0.5, 0), (1 + 2**-8, 0.5, 0)])},
                              **KID), node("l", "locator", "g", t=(0.5, 1, -2))],
     layers=[("additive", {"g": {"t": ([0, 8], [(0, 0, 0), (0, 0, 0)])}})], **TEN)
for name, channel, still, flat in (("e_merge_emptied_rotate", "r", 20, (0, 0, 0)), ("e_merge_emptied_scale", "s", 1, (1, 1, 1))):
    edge(name, [node("P", "group", None, **PN),
                node("g", "group", "P", inherit="RrSs", pre=PX5, t=(0.5, 0.5, 0), r=(5, 40, 60), s=(0.5, 1.5, 0.75),
                     ckeys=[(channel, 0, [-3, 0, 8], [still, still + 2**-8, still + 2**-7])]),
                node("l", "locator", "g", t=(0.5, 1, -2))],
         span=(2, 8), layers=[("additive", {"g": {channel: ([0, 8], [flat, flat])}}, True)], samples=[float(f) for f in range(-5, 11)])
# a live layer on another node: the conversion still reads g's own
# translate keys, level from the second, so its bake holds from frame 6
edge("e_layers_keys", [node("P", "group", None, keys=PW, **PN),
                       node("g", "group", "P", inherit="RrSs", pre=PX5, r=(0, 0, 0), s=(1, 1.5, 0.75),
                            keys={"r": ([2, 4], [(10, 0, 0), (40, 0, 0)]),
                                  "t": ([0, 4, 6, 8], [(1, 0.5, -0.5), (2, 0.5, -0.5), (2, 0.5, -0.5), (2, 0.5, -0.5)])}),
                       node("l", "locator", "g", t=(0.5, 1, -2)), node("Q", "group", None, t=(5, 0, 0)), node("n", "locator", "Q", t=(0, 1, 0))],
     span=(0, 8), layers=[("additive", {"Q": {"r": ([0, 8], [(0, 0, 0), (0, 30, 0)])}})], **MW)

# the conversion runs on the first layer alone: it empties a first-layer
# rotate or scale that stays within its tolerance whatever a live later
# layer does, which then blends over the value held; a level first layer
# under a moving layer is emptied, a level layer over a level first layer
# counts once; below a head the first layer is held, the live layers over it
GS    = dict(t=(0.5, 0.5, 0), r=(20, 40, 60), s=(0.5, 1.5, 0.75))
FIVE0 = dict(span=(2, 8), samples=[float(f) for f in range(-5, 11)])
for name, channel, still, offset in (("e_merge_flat_rotate", "r", 20, (10, 0, 0)), ("e_merge_flat_scale", "s", 1, (1.5, 1, 1))):
    edge(name, [node("P", "group", None, **PN),
                node("g", "group", "P", inherit="RrSs", pre=PX5, t=(0.5, 0.5, 0), r=(5, 40, 60), s=(0.5, 1.5, 0.75),
                     ckeys=[(channel, 0, [0, 8], [still, still])]),
                node("l", "locator", "g", t=(0.5, 1, -2))],
         span=(0, 8), layers=[("additive", {"g": {channel: ([4], [offset])}})], samples=[float(f) for f in range(-2, 11)])
edge("e_merge_emptied_live", [node("P", "group", None, **PN),
                              node("g", "group", "P", inherit="RrSs", pre=PX5, t=(0.5, 0.5, 0), r=(5, 40, 60), s=(0.5, 1.5, 0.75),
                                   ckeys=[("s", 0, [-3, 0, 8], [1, 1 + 2**-8, 1 + 2**-7])]),
                              node("l", "locator", "g", t=(0.5, 1, -2))],
     layers=[("additive", {"g": {"s": ([0, 8], [(1, 1, 1), (1.5, 1, 1)])}})], **FIVE0)
edge("e_merge_level", [node("P", "group", None, **PN),
                       node("g", "group", "P", inherit="RrSs", pre=PX5, ckeys=[("s", 0, [-3, 0, 8], [1.0, 1.002, 1.004])], **GS),
                       node("l", "locator", "g", t=(0.5, 1, -2))],
     layers=[("additive", {"g": {"s": ([-3, 8], [(1.5, 1, 1), (1.5, 1, 1)])}})], **FIVE0)
edge("e_merge_level_root", [node("g", "group", None, inherit="Rrs", ckeys=[("s", 0, [-3, 0, 8], [1.0, 1.002, 1.004])], **GS),
                            node("l", "locator", "g", t=(0.5, 1, -2)), node("h", "group", "g", inherit="Rrs", t=(0, 1, 0), s=(1, 2, 3))],
     axes="zup", layers=[("additive", {"g": {"s": ([-3, 8], [(1.5, 1, 1), (1.5, 1, 1)])}})], **FIVE0)
edge("e_merge_moving", [node("P", "group", None, **PN),
                        node("g", "group", "P", inherit="RrSs", pre=PX5, ckeys=[("s", 0, [-3, 0, 8], [1.0, 1.004, 1.008])], **GS),
                        node("l", "locator", "g", t=(0.5, 1, -2))],
     layers=[("additive", {"g": {"s": ([-3, 8], [(1.25, 1, 1), (1.75, 1, 1)])}})], **FIVE0)
FW = list(range(-12, 17))
edge("e_merge_head_live", [node("P", "group", None, keys={"s": (FW, [(1 + 0.05 * (f + 12), 2, 3 - 0.05 * (f + 12)) for f in FW])}, **PN),
                           node("g", "group", "P", inherit="RrSs", pre=PX5,
                                keys={"r": ([-2.5, 0.5, 3.5], UF), "s": ([-1, 4], [(1, 1.5, 0.75), (1.25, 1.5, 0.75)])}, **KID)] + tail("g"),
     span=(-6, 8), layers=[("additive", {"g": {"s": ([-6, 8], [(1, 1, 1), (1.5, 1.25, 1)])}})], samples=[float(f) for f in range(-8, 13)])
edge("e_merge_head_held", [node("P", "group", None, keys={"s": (FW, [(1 + 0.05 * (f + 12), 2, 3 - 0.05 * (f + 12)) for f in FW])}, **PN),
                           node("g", "group", "P", inherit="RrSs", pre=PX5,
                                keys={"r": ([-2.5, 0.5, 3.5], UF), "s": ([-2.5, 4], [(1, 1.5, 0.75), (1.75, 1.5, 0.75)])}, **KID)] + tail("g"),
     span=(-6, 8), layers=[("additive", {"g": {"s": ([-6, 8], [(1, 1, 1), (1.5, 1.25, 1)])}})], samples=[float(f) for f in range(-8, 13)])
edge("e_merge_head_rotate_live", [node("P", "group", None, keys={"s": (FW, [(1 + 0.05 * (f + 12), 2, 3 - 0.05 * (f + 12)) for f in FW])}, **PN),
                                  node("g", "group", "P", inherit="RrSs", pre=PX5, keys={"r": ([-2.5, 0.5, 3.5], UF)}, **KID)] + tail("g"),
     span=(-6, 8), layers=[("additive", {"g": {"r": ([-6, 8], [(0, 0, 0), (30, 10, 0)])}})], samples=[float(f) for f in range(-8, 13)])

# a skinned mesh under a camera root: against Maya's world of the camera,
# mirrored or baked; held on a clip's frames where it has no curve, and
# where it has one the static channels but the ones it keys, unless it
# bakes; a mesh with no skin read as any node
CAM3 = dict(CAM, s=(1, 2, 3))
R0J  = node("r0", "joint", None, t=(0, 1, 0))
edge("e_camera_mirror", [R0J, node("C", "camera", None, **dict(CAM, s=(-1, 2, 3))), skinned("g", "C")] + tail("g"),  axes="zup", **TEN)
edge("e_camera_neg2",   [R0J, node("C", "camera", None, **dict(CAM, s=(-1, -2, 3))), skinned("g", "C")] + tail("g"), axes="xup", **TEN)
edge("e_camera_rrs",    [R0J, node("C", "camera", None, inherit="Rrs", **CAM3), skinned("g", "C")] + tail("g"),      axes="zup", **TEN)
edge("e_camera_rrs_keyed", [R0J, node("C", "camera", None, inherit="Rrs", keys={"r": ([0, 8], [(10, 30, -20), (40, 30, -20)])}, **CAM3),
                            skinned("g", "C")] + tail("g"), axes="xup", **TEN)
edge("e_camera_scale_keyed", [R0J, node("C", "camera", None, keys={"s": ([0, 8], [(1, 2, 3), (2, 1, 3)])}, **CAM3), skinned("g", "C")] + tail("g"),
     axes="zup", **TEN)
edge("e_camera_mesh_rotate", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C", keys={"r": ([0, 8], [(20, 40, 60), (50, 45, 60)])})] + tail("g"),
     axes="zup", **TEN)
edge("e_camera_mesh_scale", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C", keys={"s": ([0, 8], [(1, 1.5, 0.75), (2, 1, 0.5)])})] + tail("g"),
     axes="zup", **TEN)
edge("e_camera_mesh_two", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C"),
                           skinned("g2", "C", keys={"t": ([0, 8], [(1, 0.5, 0), (2, 0.5, 1)])})] + tail("g"), axes="xup", **TEN)
edge("e_camera_mesh_baked", [R0J, node("C", "camera", None, **CAM),
                             skinned("g", "C", "Rrs", keys={"r": ([0, 4, 8], [(20, 40, 60), (35, 40, 60), (50, 55, 60)])})] + tail("g"), axes="zup", **TEN)
edge("e_camera_noskin", [R0J, node("C", "camera", None, **CAM3), node("g", "mesh", "C", points=MESH, **KID)] + tail("g"), axes="zup", **TEN)
# a curve keys its own axis alone: the others hold the static values, and
# the mesh turns about its pivots by the rotate and scale it then holds
edge("e_camera_axis_rotate", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C", ckeys=[("r", 0, [0, 8], [20, 50])])] + tail("g"),
     axes="zup", **TEN)
edge("e_camera_axis_translate", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C", ckeys=[("t", 1, [0, 8], [0.5, 1.5])])] + tail("g"),
     axes="xup", **TEN)
edge("e_camera_axis_scale", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C", ckeys=[("s", 2, [0, 8], [0.75, 1.5])])] + tail("g"),
     axes="zup", **TEN)
edge("e_camera_mesh_pivots", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C", piv=PIV, keys={"r": ([0, 8], [(20, 40, 60), (50, 45, 60)])})]
     + tail("g"), axes="zup", **TEN)
# the axes no curve keys hold Maya's own solve of the static rotation, here
# the other solution, whose absolute angles sum the smaller: under a camera
# scaled unevenly, and under one scaled 1, no shear anywhere
edge("e_camera_axis_branch", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C", r=(124.9, 158.6, 48.5), ckeys=[("r", 0, [0, 8], [124.9, 154.9])])]
     + tail("g"), axes="zup", **TEN)
edge("e_camera_axis_branch_even", [R0J, node("C", "camera", None, **CAM),
                                   skinned("g", "C", r=(172.1, 120.7, 115.3), s=(1, 1, 1), ckeys=[("r", 0, [0, 8], [172.1, 202.1])])]
     + tail("g"), axes="zup", **TEN)
# an outer axis of the static rotation at a half turn, +180 read as -180, so
# the other solution's sum is the smaller: a mesh turned (0, 90, 0) under a
# camera scaled 1
edge("e_camera_axis_half", [R0J, node("C", "camera", None, **CAM),
                            skinned("g", "C", r=(0, 90, 0), s=(1, 1, 1), ckeys=[("r", 0, [0, 8], [0, 30])])]
     + tail("g"), axes="zup", **TEN)
# one axis keyed on a mesh with a pre rotation, in xyz, the angles a(f)'s; with
# a post rotation, held past the rotate axis; in another order, composed in it;
# with pivots, keyed in scale; a later layer keying another axis, muted: the
# channel merged whole
edge("e_camera_axis_pre", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C", pre=PX5, ckeys=[("r", 0, [0, 8], [20, 50])])] + tail("g"),
     axes="zup", **TEN)
edge("e_camera_axis_post", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C", post=(0, 0, 30), ckeys=[("r", 0, [0, 8], [20, 50])])]
     + tail("g"), axes="zup", **TEN)
edge("e_camera_axis_order", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C", order="zxy", ckeys=[("r", 0, [0, 8], [20, 50])])]
     + tail("g"), axes="xup", **TEN)
edge("e_camera_axis_pivots", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C", piv=PIV, ckeys=[("s", 2, [0, 8], [0.75, 1.5])])]
     + tail("g"), axes="zup", **TEN)
edge("e_camera_axis_later", [R0J, node("C", "camera", None, **CAM3), skinned("g", "C", ckeys=[("r", 0, [0, 8], [20, 50])])] + tail("g"),
     axes="zup", layers=[("additive", {"g": {"ckeys": [("r", 1, [2, 6], [5, 35])]}}, True)], **TEN)

# a camera under a joint is no joint in Maya's import, whatever its pre and
# post rotation
edge("e_bare_camera", [node("J", "joint", None, **JS), node("N", "camera", "J", **dict(NK, pre=PX5, post=(0, 15, 0)))] + below("N"), **NJ)


def write_layout():
    """_fbx_import_files.write_layout, by path: no cgmath is imported here"""
    path   = os.path.join(HERE, "..", "_tests", "_fbx_import_files.py")
    spec   = importlib.util.spec_from_file_location("_fbx_import_files", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_layout


def imported(path, animation):
    cmds.file(new=True, force=True)
    mel.eval("FBXResetImport")
    if not animation:
        mel.eval('FBXProperty "Import|IncludeGrp|Animation" -v false')
    mel.eval('FBXImport -f "%s"' % path)
    if not animation:
        mel.eval('FBXProperty "Import|IncludeGrp|Animation" -v true')


def world(name):
    return np.array(cmds.xform(name, q=True, ws=True, m=True)).reshape(4, 4)[:, :3].ravel()


def mesh_points(name):
    shapes = cmds.listRelatives(name, shapes=True, type="mesh", fullPath=True) or []
    if not shapes:
        return None
    selection = om.MSelectionList()
    selection.add(shapes[0])
    points = om.MFnMesh(selection.getDagPath(0)).getPoints(om.MSpace.kObject)
    return np.array([[p.x, p.y, p.z] for p in points])


def record(name, fx, path, arrays):
    """Maya's two imports of ``path`` into ``arrays``"""
    frames = fx.get("samples", SAMPLES)
    if name not in HALF:
        frames = [f for f in frames if f == int(f)]
    parents = {n["parent"] for n in fx["nodes"]}
    keep    = [n["name"] for n in fx["nodes"] if n["name"] not in parents or (n["inherit"] != "RSrs" and n["kind"] != "joint")]

    imported(path, True)
    keep   = [n for n in keep if cmds.objExists(n)]
    rows   = {n: [] for n in keep}
    shears = {n: [] for n in keep}
    for frame in frames:
        cmds.currentTime(frame)
        for n in keep:
            rows[n].append(world(n))
            if name in SHEAR:
                shears[n].append(cmds.getAttr(n + ".shear")[0])

    arrays[name + "|layout"] = np.frombuffer(json.dumps(fx, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8)
    arrays[name + "|frames"] = np.array(frames, dtype=float)
    for n in keep:
        arrays[name + "|" + n] = np.array(rows[n])
        if name in SHEAR:
            arrays[name + "|h|" + n] = np.array(shears[n])
    if "x_me_" in name:
        for spec in fx["nodes"]:
            points = mesh_points(spec["name"]) if spec["kind"] == "mesh" else None
            if points is not None:
                arrays[name + "|p|" + spec["name"]] = points

    imported(path, False)
    for n in keep:
        arrays[name + "|s|" + n] = world(n)


def save(path, arrays):
    """np.savez_compressed with a fixed date on every entry."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in arrays.items():
            info               = zipfile.ZipInfo(key + ".npy", date_time=DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(value), allow_pickle=False)


def main(out):
    cmds.loadPlugin("fbxmaya", quiet=True)
    write = write_layout()
    arrays = {
        "maya_version":     np.array(cmds.about(installedVersion=True)),
        "maya_api_version": np.array(cmds.about(apiVersion=True), dtype=np.int64),
        "names":            np.array(list(FILES)),
        "limits":           np.full(len(FILES), np.nan),
        "edges":            np.array(list(EDGES)),
    }
    with tempfile.TemporaryDirectory() as folder:
        for name, fx in list(FILES.items()) + list(EXTRA.items()) + list(EDGES.items()):
            path = os.path.join(folder, name + ".fbx").replace(os.sep, "/")
            write(path, json.loads(json.dumps(fx)))
            record(name, fx, path, arrays)
    cmds.file(new=True, force=True)
    save(out, arrays)
    print(f"{out}: {os.path.getsize(out)} bytes, {len(FILES)} files and {len(EDGES)} edges")


def limits(root, out):
    """each file's position limit, read through the cgmath of ``root``"""
    sys.path.insert(0, root)
    from cgmath._tests.test_fbx_maya_import import position_gaps

    with np.load(out) as data:
        arrays = {key: data[key] for key in data.files}
    with tempfile.TemporaryDirectory() as folder:
        for index, name in enumerate(arrays["names"].tolist()):
            arrays["limits"][index] = position_gaps(arrays, name, folder)
    save(out, arrays)
    print(f"{out}: {os.path.getsize(out)} bytes, limits up to {arrays['limits'].max()!r}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--limits"]:
        limits(os.path.abspath(args[1]), os.path.abspath(args[2] if len(args) > 2 else OUT))
    else:
        main(os.path.abspath(args[0] if args else OUT))
    maya.standalone.uninitialize()
