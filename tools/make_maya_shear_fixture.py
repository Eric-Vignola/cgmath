"""Record what Maya does with shear, for cgmath's shear tests.

usage (Windows, from the repository root):

    set MAYA_DISABLE_CER=1
    "C:/Program Files/Autodesk/Maya2025/bin/mayapy.exe" tools/make_maya_shear_fixture.py [out.npz]

The archive goes to _tests/test_assets/maya_shear.npz unless another path is
given. Every input comes from the fixed seed SEED and the zip entries carry a
fixed date, so a second run with the same Maya writes the same bytes. Nothing
here imports cgmath: the inputs are built with numpy and set through Maya's
own commands, so the code under test has no hand in its own reference.

Maya's rows are row vectors: a point is p M, translation in row 3, and every
matrix below is read with getAttr row by row. Pivots stay at 0. Channels are
stored as Maya reads them back after they were set, angles in degrees; a
rotate Maya returns is on Maya's own Euler branch, so compare rotations as
matrices rather than angles.

Scale kinds (scale_kinds, the index in *_scale_kind): plain (each axis in
0.3..2.5), uniform, mirror_x, mirror_y, mirror_z (that axis negative),
mirror_two (two axes negative), mirror_xyz (all three negative), and
near_zero_1e-4, near_zero_1e-8, near_zero_1e-12 (one axis at that size, either
sign, the others plain). Shear kinds (shear_kinds): zero, tiny (each entry
+-1e-13), plain (each entry in -1..1), large (each entry in -3..3). The near
zero scales put entries near 1e12 in the matrix of a segment scale
compensated child: compare those relatively.

Two records:

compose: chains of three nodes (root, middle, leaf), every pattern of
    transform and joint along the chain, rotate orders, scale and shear kinds
    in turn, rotate axis on about half the nodes, a joint orient on every
    joint, segment scale compensate on about half the joints. Every joint
    with a parent has its inverseScale connected from the parent's scale, a
    transform parent's too, so inverseScale is the parent's scale wherever
    there is a parent.

    compose_node_type                 (N,)       0 transform, 1 joint
    compose_parent                    (N,)       the parent's row, -1 at a root
    compose_scale_kind                (N,)       index into scale_kinds
    compose_shear_kind                (N,)       index into shear_kinds
    compose_translate                 (N, 3)
    compose_rotate                    (N, 3)
    compose_scale                     (N, 3)
    compose_shear                     (N, 3)     shearXY, shearXZ, shearYZ
    compose_rotate_axis               (N, 3)
    compose_joint_orient              (N, 3)     0 on a transform
    compose_rotate_order              (N,)       0..5 = xyz yzx zxy xzy yxz zyx
    compose_segment_scale_compensate  (N,)       False on a transform
    compose_inverse_scale             (N, 3)     as Maya reads it; 1 on roots
                                                 and transforms
    compose_matrix                    (N, 4, 4)  node.matrix
    compose_world_matrix              (N, 4, 4)  node.worldMatrix[0]
    compose_decompose_scale ...       (N, 3)     decomposeMatrix of
      _shear, _rotate, _translate                compose_matrix, inputRotateOrder
                                                 the node's rotate order
    compose_world_decompose_scale ... (N, 3)     the same of compose_world_matrix

split: a target matrix set with xform -matrix (object space) or
    xform -worldSpace -matrix on a transform or a joint, in that order of
    setups, each rotate order in turn. Targets are S Sh R T from random
    channels (a scale and a shear kind, a rotation in a random order). The
    node has a rotate axis on about half the records; a joint has a joint
    orient, segment scale compensate on about half, and its own shear before
    the xform on a third. Object space joints sit at the world or, on every
    other record, under a scaled joint; world space nodes sit under a
    transform or a joint parent holding scale, shear, rotate, translate (and
    a joint orient on a joint parent). A joint with a parent has its
    inverseScale connected from the parent's scale.

    split_node_type                   (M,)       0 transform, 1 joint
    split_space                       (M,)       0 xform -matrix, 1 xform -worldSpace -matrix
    split_scale_kind, split_shear_kind (M,)      the target's kinds
    split_target                      (M, 4, 4)  the matrix given to xform
    split_has_parent                  (M,)       bool
    split_parent_world                (M, 4, 4)  the parent's worldMatrix, identity at the world
    split_parent_scale                (M, 3)     the parent's scale, 1 at the world
    split_rotate_order                (M,)
    split_segment_scale_compensate    (M,)
    split_before_translate ...        (M, 3)     the node's channels before the
      _rotate, _scale, _shear,                   xform, as Maya reads them
      _rotate_axis, _joint_orient
    split_after_translate ...         (M, 3)     the same after the xform
      _rotate, _scale, _shear,
      _rotate_axis, _joint_orient
    split_inverse_scale               (M, 3)     the joint's inverseScale, 1 on a transform
    split_matrix_after                (M, 4, 4)  node.matrix after the xform
    split_world_after                 (M, 4, 4)  node.worldMatrix[0] after the xform
    split_decompose_scale ...         (M, 3)     decomposeMatrix of split_target,
      _shear, _rotate, _translate                inputRotateOrder the node's

scale_kinds, shear_kinds, seed, maya_version and maya_api_version say how it
was made and which Maya wrote it.

What Maya 2025 did with these inputs when this was written, for whoever
picks records for a test (measured, not checked on writing):
    - node.matrix applies every shear set, tiny ones included, but
      decomposeMatrix and xform -matrix write 0 for each shear entry whose
      scale times shear is about 1e-10 or less: the tiny kind comes back 0.
    - a scale below about 1e-10 is lossy in Maya's split: the
      near_zero_1e-12 records do not recompose (1e-4 and 1e-8 do, on
      transforms).
    - xform -matrix on a joint writes no shear: the joint keeps the shear it
      had, so split_after_shear equals split_before_shear on every joint.
    - on a segment scale compensated joint whose parent's scale is uneven or
      mirrored, xform misses even a target the joint can hold.
"""
import os
import sys
import zipfile

import numpy as np

import maya.standalone

maya.standalone.initialize(name="python")

import maya.cmds as cmds  # noqa: E402

HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(HERE, "..", "_tests", "test_assets", "maya_shear.npz")

SEED   = 20261030
CHAINS = 72  # compose: 216 nodes
SPLITS = 6   # split: records per setup and rotate order, 4 setups x 6 orders

SCALE_KINDS = ["plain", "uniform", "mirror_x", "mirror_y", "mirror_z", "mirror_two", "mirror_xyz",
               "near_zero_1e-4", "near_zero_1e-8", "near_zero_1e-12"]
SHEAR_KINDS = ["zero", "tiny", "plain", "large"]
ORDERS      = ["xyz", "yzx", "zxy", "xzy", "yxz", "zyx"]

# a zip entry's date is the time of writing unless given
DATE = (1980, 1, 1, 0, 0, 0)

# the grid the random inputs sit on: exact in binary, and compresses well
GRID = 2.0**10


def on_grid(values):
    return np.round(np.asarray(values, dtype=float) * GRID) / GRID


def scale_of(rng, kind):
    name = SCALE_KINDS[kind]
    s    = on_grid(rng.uniform(0.3, 2.5, 3))
    if name == "uniform":
        s[:] = s[0]
    elif name.startswith("mirror_") and name[-1] in "xyz" and len(name) == 8:
        s["xyz".index(name[-1])] *= -1
    elif name == "mirror_two":
        s[rng.permutation(3)[:2]] *= -1
    elif name == "mirror_xyz":
        s *= -1
    elif name.startswith("near_zero_"):
        s[rng.integers(3)] = float(name[len("near_zero_"):]) * rng.choice((-1.0, 1.0))
    return s


def shear_of(rng, kind):
    name = SHEAR_KINDS[kind]
    if name == "zero":
        return np.zeros(3)
    if name == "tiny":
        return 1e-13 * rng.choice((-1.0, 1.0), 3)
    return on_grid(rng.uniform(-1.0, 1.0, 3) * (3.0 if name == "large" else 1.0))


def shear_kind(rng):
    return int(rng.choice(4, p=(0.2, 0.1, 0.55, 0.15)))


def angles(rng, size=180.0):
    return on_grid(rng.uniform(-size, size, 3))


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def rotation(degrees, order):
    """row vectors: rotate order "abc" applies a, then b, then c"""
    out = np.eye(3)
    for axis, a in zip(ORDERS[order], np.radians(degrees)[["xyz".index(c) for c in ORDERS[order]]]):
        out = out @ {"x": rx, "y": ry, "z": rz}[axis](a)
    return out


def target(rng):
    """S Sh R T from random channels, with its kinds"""
    sk, hk = int(rng.integers(len(SCALE_KINDS))), shear_kind(rng)
    s, sh = scale_of(rng, sk), shear_of(rng, hk)
    Sh        = np.array([[1, 0, 0], [sh[0], 1, 0], [sh[1], sh[2], 1]])
    M         = np.eye(4)
    M[:3, :3] = np.diag(s) @ Sh @ rotation(angles(rng), int(rng.integers(6)))
    M[3, :3]  = on_grid(rng.uniform(-5.0, 5.0, 3))
    return M, sk, hk


def set3(node, attr, v):
    cmds.setAttr(f"{node}.{attr}", *[float(x) for x in v], type="double3")


def get3(node, attr):
    return list(cmds.getAttr(f"{node}.{attr}")[0])


def getm(plug):
    return np.array(cmds.getAttr(plug), dtype=float).reshape(4, 4)


def is_joint(node):
    return cmds.nodeType(node) == "joint"


def read(node):
    """translate, rotate, scale, shear, rotate axis, joint orient as Maya reads them"""
    out = [get3(node, a) for a in ("translate", "rotate", "scale", "shear", "rotateAxis")]
    out.append(get3(node, "jointOrient") if is_joint(node) else [0.0, 0.0, 0.0])
    return out


class Decompose:
    """one decomposeMatrix node per rotate order"""

    def __init__(self):
        self.nodes = []
        for k in range(6):
            node = cmds.createNode("decomposeMatrix", name=f"decompose_{ORDERS[k]}")
            cmds.setAttr(node + ".inputRotateOrder", k)
            self.nodes.append(node)

    def __call__(self, M, order):
        node = self.nodes[order]
        cmds.setAttr(node + ".inputMatrix", *[float(x) for x in np.asarray(M).reshape(16)], type="matrix")
        return [get3(node, a) for a in ("outputScale", "outputShear", "outputRotate", "outputTranslate")]


def record_compose(rng, decompose):
    rows = {k: [] for k in ("type", "parent", "sk", "hk", "order", "ssc", "nodes")}
    for chain in range(CHAINS):
        above = None
        for k in range(3):
            kind  = "joint" if (chain >> k) & 1 else "transform"
            node  = cmds.createNode(kind, parent=above) if above else cmds.createNode(kind)
            node  = cmds.ls(node, long=True)[0]
            order = (chain + k) % 6
            sk    = (3 * chain + k) % len(SCALE_KINDS)
            hk    = shear_kind(rng)
            cmds.setAttr(node + ".rotateOrder", order)
            set3(node, "translate",  on_grid(rng.uniform(-5.0, 5.0, 3)))
            set3(node, "rotate",     angles(rng))
            set3(node, "scale",      scale_of(rng, sk))
            set3(node, "shear",      shear_of(rng, hk))
            set3(node, "rotateAxis", angles(rng, 90.0) if rng.random() < 0.5 else np.zeros(3))
            ssc = False
            if kind == "joint":
                set3(node, "jointOrient", angles(rng, 90.0))
                ssc = bool(rng.random() < 0.5)
                cmds.setAttr(node + ".segmentScaleCompensate", ssc)
                if above:
                    cmds.connectAttr(above + ".scale", node + ".inverseScale", force=True)
            rows["type"].append(int(kind == "joint"))
            rows["parent"].append(len(rows["nodes"]) - 1 if above else -1)
            rows["sk"].append(sk)
            rows["hk"].append(hk)
            rows["order"].append(order)
            rows["ssc"].append(ssc)
            rows["nodes"].append(node)
            above = node

    channels, inverse, local, world, dec, wdec = [], [], [], [], [], []
    for node, order in zip(rows["nodes"], rows["order"]):
        channels.append(read(node))
        inverse.append(get3(node, "inverseScale") if is_joint(node) else [1.0, 1.0, 1.0])
        local.append(getm(node + ".matrix"))
        world.append(getm(node + ".worldMatrix[0]"))
        dec.append(decompose(local[-1], order))
        wdec.append(decompose(world[-1], order))
    channels, dec, wdec = np.array(channels), np.array(dec), np.array(wdec)
    cmds.delete([n for n, p in zip(rows["nodes"], rows["parent"]) if p == -1])

    out = {
        "compose_node_type": np.array(rows["type"], dtype=np.int8),
        "compose_parent": np.array(rows["parent"], dtype=np.int32),
        "compose_scale_kind": np.array(rows["sk"], dtype=np.int8),
        "compose_shear_kind": np.array(rows["hk"], dtype=np.int8),
        "compose_rotate_order": np.array(rows["order"], dtype=np.int8),
        "compose_segment_scale_compensate": np.array(rows["ssc"], dtype=bool),
        "compose_inverse_scale": np.array(inverse),
        "compose_matrix": np.array(local),
        "compose_world_matrix": np.array(world),
    }
    for i, name in enumerate(("translate", "rotate", "scale", "shear", "rotate_axis", "joint_orient")):
        out["compose_" + name] = channels[:, i]
    for i, name in enumerate(("scale", "shear", "rotate", "translate")):
        out["compose_decompose_" + name]       = dec[:, i]
        out["compose_world_decompose_" + name] = wdec[:, i]
    return out


def record_split(rng, decompose):
    keys = ("type", "space", "sk", "hk", "target", "has_parent", "parent_world", "parent_scale", "order", "ssc",
            "before", "after", "inverse", "matrix", "world", "dec")
    rows   = {k: [] for k in keys}
    setups = (("transform", 0), ("joint", 0), ("transform", 1), ("joint", 1))
    for kind, space in setups:
        for order in range(6):
            for i in range(SPLITS):
                parent = None
                if space == 1 or (kind == "joint" and i % 2):
                    pkind  = "joint" if (space == 0 or rng.random() < 0.5) else "transform"
                    parent = cmds.createNode(pkind)
                    set3(parent, "translate", on_grid(rng.uniform(-3.0, 3.0, 3)))
                    set3(parent, "rotate",    angles(rng))
                    set3(parent, "scale",     scale_of(rng, int(rng.integers(7))))  # no near zero parent
                    set3(parent, "shear",     shear_of(rng, 2) if space == 1 else np.zeros(3))
                    if pkind == "joint":
                        set3(parent, "jointOrient", angles(rng, 90.0))
                        cmds.setAttr(parent + ".segmentScaleCompensate", False)
                node = cmds.createNode(kind, parent=parent) if parent else cmds.createNode(kind)
                node = cmds.ls(node, long=True)[0]
                cmds.setAttr(node + ".rotateOrder", order)
                set3(node, "translate",  on_grid(rng.uniform(-3.0, 3.0, 3)))
                set3(node, "rotate",     angles(rng))
                set3(node, "scale",      on_grid(rng.uniform(0.5, 2.0, 3)))
                set3(node, "rotateAxis", angles(rng, 90.0) if rng.random() < 0.5 else np.zeros(3))
                ssc = False
                if kind == "joint":
                    set3(node, "jointOrient", angles(rng, 90.0))
                    set3(node, "shear", shear_of(rng, 2) if rng.random() < 1 / 3 else np.zeros(3))
                    ssc = bool(rng.random() < 0.5)
                    cmds.setAttr(node + ".segmentScaleCompensate", ssc)
                    if parent:
                        cmds.connectAttr(parent + ".scale", node + ".inverseScale", force=True)
                else:
                    set3(node, "shear", shear_of(rng, 2))
                M, sk, hk = target(rng)
                before = read(node)
                flat   = [float(x) for x in M.reshape(16)]
                if space == 0:
                    cmds.xform(node, matrix=flat)
                else:
                    cmds.xform(node, worldSpace=True, matrix=flat)
                rows["type"].append(int(kind == "joint"))
                rows["space"].append(space)
                rows["sk"].append(sk)
                rows["hk"].append(hk)
                rows["target"].append(M)
                rows["has_parent"].append(parent is not None)
                rows["parent_world"].append(getm(parent + ".worldMatrix[0]") if parent else np.eye(4))
                rows["parent_scale"].append(get3(parent, "scale") if parent else [1.0, 1.0, 1.0])
                rows["order"].append(order)
                rows["ssc"].append(ssc)
                rows["before"].append(before)
                rows["after"].append(read(node))
                rows["inverse"].append(get3(node, "inverseScale") if kind == "joint" else [1.0, 1.0, 1.0])
                rows["matrix"].append(getm(node + ".matrix"))
                rows["world"].append(getm(node + ".worldMatrix[0]"))
                rows["dec"].append(decompose(M, order))
                cmds.delete(parent or node)

    before, after, dec = np.array(rows["before"]), np.array(rows["after"]), np.array(rows["dec"])
    out = {
        "split_node_type": np.array(rows["type"], dtype=np.int8),
        "split_space": np.array(rows["space"], dtype=np.int8),
        "split_scale_kind": np.array(rows["sk"], dtype=np.int8),
        "split_shear_kind": np.array(rows["hk"], dtype=np.int8),
        "split_target": np.array(rows["target"]),
        "split_has_parent": np.array(rows["has_parent"], dtype=bool),
        "split_parent_world": np.array(rows["parent_world"]),
        "split_parent_scale": np.array(rows["parent_scale"]),
        "split_rotate_order": np.array(rows["order"], dtype=np.int8),
        "split_segment_scale_compensate": np.array(rows["ssc"], dtype=bool),
        "split_inverse_scale": np.array(rows["inverse"]),
        "split_matrix_after": np.array(rows["matrix"]),
        "split_world_after": np.array(rows["world"]),
    }
    for i, name in enumerate(("translate", "rotate", "scale", "shear", "rotate_axis", "joint_orient")):
        out["split_before_" + name] = before[:, i]
        out["split_after_" + name]  = after[:, i]
    for i, name in enumerate(("scale", "shear", "rotate", "translate")):
        out["split_decompose_" + name] = dec[:, i]
    return out


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
    cmds.currentUnit(angle="degree", linear="centimeter")
    cmds.loadPlugin("matrixNodes", quiet=True)
    decompose = Decompose()

    arrays = {
        "maya_version":     np.array(cmds.about(installedVersion=True)),
        "maya_api_version": np.array(cmds.about(apiVersion=True), dtype=np.int64),
        "seed":             np.array(SEED, dtype=np.int64),
        "scale_kinds":      np.array(SCALE_KINDS),
        "shear_kinds":      np.array(SHEAR_KINDS),
    }
    arrays.update(record_compose(np.random.default_rng([SEED, 1]), decompose))
    arrays.update(record_split(np.random.default_rng([SEED, 2]), decompose))
    save(out, arrays)
    print(f"{out}: {os.path.getsize(out)} bytes")
    for key, value in arrays.items():
        print(f"    {key:34} {str(value.dtype):8} {value.shape}")


if __name__ == "__main__":
    main()
    maya.standalone.uninitialize()
