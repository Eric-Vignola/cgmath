"""Record five Maya NURBS surfaces with uneven knots, for _tests/test_geometry.py
(TestBSplinePatchKnotRange).

usage (Windows, from the repository root):

    set MAYA_DISABLE_CER=1
    "C:/Program Files/Autodesk/Maya2025/bin/mayapy.exe" tools/make_maya_bspline_surface_fixture.py [out.json]

The json goes to _tests/test_assets/test_geometry.bspline_maya_surfaces.json
unless another path is given. Nothing is random, so a second run with the
same Maya writes the same bytes. Nothing here imports cgmath: the surfaces
are built and read with Maya's own commands and API, so the code under test
has no hand in its own reference.

The surfaces:

    cylinder_inserted_knot        cmds.cylinder(spans=2), then
                                  insertKnotSurface at v=0.37
    loft_chord_length             cmds.loft through circles of radius 1, 3, 1
                                  at z = 0, 2, 5, uniform=False
    loft_degree1_chord_length     the same with degree=1, radius 1, 2, 2.5,
                                  1.5 at z = 0, 0.5, 2, 2.2
    own_knots_degree2_by_degree1  cmds.surface(degreeU=2, degreeV=1,
                                  knotU=[5, 5, 5.5, 7, 8, 8],
                                  knotV=[0, 0.4, 1.5, 2])
    torus_inserted_knots          cmds.torus(sections=8, spans=4), then
                                  insertKnotSurface at u=0.6 and v=2.3

Each one read with MFnNurbsSurface, in object space:

    degree_u, degree_v, periodic_u, periodic_v
    points                (nu, nv, 3) CVs, u-major, the periodic wrap trimmed
    maya_knots_u, _v      knotsInU(), knotsInV()
    maya_domain_u, _v     knotDomainInU, knotDomainInV
    params                (36, 2) a 6 x 6 grid of each domain, v fastest
    maya_points           getDerivativesAtParam() there: the point,
    maya_derivatives_u    dS/du
    maya_derivatives_v    and dS/dv

queried in that order. On a knot where a direction is only C0, Maya's
derivative is the side of its previous query: along this ascending sweep,
the span below's. Every value is rounded to 12 decimals; each goes on one
line, as in the curves' fixture beside it.
"""
import json
import os
import sys

import numpy as np

import maya.standalone

maya.standalone.initialize(name="python")

import maya.api.OpenMaya as om  # noqa: E402
import maya.cmds as cmds  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "_tests", "test_assets", "test_geometry.bspline_maya_surfaces.json")

SOURCE = (
    "cmds.cylinder / loft / surface / torus with insertKnotSurface, read with MFnNurbsSurface; "
    "see TestBSplinePatchKnotRange"
)


def rounded(values):
    """values as nested lists of floats, rounded to 12 decimals"""
    return np.round(np.asarray(values, dtype=float), 12).tolist()


def shape_of(node):
    """the node's shape, or the node itself"""
    shapes = cmds.listRelatives(node, shapes=True, fullPath=True)
    return shapes[0] if shapes else node


def read(node):
    """one surface's record: the docstring's keys, in its order"""
    selection = om.MSelectionList()
    selection.add(shape_of(node))
    fn         = om.MFnNurbsSurface(selection.getDagPath(0))
    periodic_u = fn.formInU == om.MFnNurbsSurface.kPeriodic
    periodic_v = fn.formInV == om.MFnNurbsSurface.kPeriodic
    nu, nv = fn.numCVsInU, fn.numCVsInV
    du, dv = fn.degreeInU, fn.degreeInV
    grid = np.array(fn.cvPositions(om.MSpace.kObject))[:, :3].reshape(nu, nv, 3)
    if periodic_u:
        grid = grid[: nu - du]
    if periodic_v:
        grid = grid[:, : nv - dv]

    domain_u, domain_v = fn.knotDomainInU, fn.knotDomainInV
    params, points, along_u, along_v = [], [], [], []
    for u in np.linspace(domain_u[0], domain_u[1], 6):
        for v in np.linspace(domain_v[0], domain_v[1], 6):
            point, d_u, d_v = fn.getDerivativesAtParam(float(u), float(v), om.MSpace.kObject)[:3]
            params.append([u, v])
            points.append(list(point)[:3])
            along_u.append(list(d_u))
            along_v.append(list(d_v))

    return {
        "degree_u":           du,
        "degree_v":           dv,
        "periodic_u":         bool(periodic_u),
        "periodic_v":         bool(periodic_v),
        "points":             rounded(grid),
        "maya_knots_u":       rounded(fn.knotsInU()),
        "maya_knots_v":       rounded(fn.knotsInV()),
        "maya_domain_u":      rounded(domain_u),
        "maya_domain_v":      rounded(domain_v),
        "params":             rounded(params),
        "maya_points":        rounded(points),
        "maya_derivatives_u": rounded(along_u),
        "maya_derivatives_v": rounded(along_v),
    }


def circles(radii, heights):
    """circles about z, one per radius, frozen at their heights"""
    out = []
    for radius, z in zip(radii, heights):
        circle = cmds.circle(radius=radius, normal=(0, 0, 1), constructionHistory=False)[0]
        cmds.move(0, 0, z, circle)
        cmds.makeIdentity(circle, apply=True, translate=True)
        out.append(circle)
    return out


def record():
    """the five surfaces' records, by name, each read as soon as it is built"""
    surfaces = {}

    def add(name, node):
        surfaces[name] = read(node)

    cylinder = cmds.cylinder(spans=2, constructionHistory=False)[0]
    cmds.insertKnotSurface(shape_of(cylinder) + ".v[0.37]", constructionHistory=False, replaceOriginal=True)
    add("cylinder_inserted_knot", cylinder)

    loft = cmds.loft(*circles([1, 3, 1], [0, 2, 5]), uniform=False, constructionHistory=False)[0]
    add("loft_chord_length", loft)

    loft = cmds.loft(
        *circles([1, 2, 2.5, 1.5], [0, 0.5, 2, 2.2]), degree=1, uniform=False, constructionHistory=False
    )[0]
    add("loft_degree1_chord_length", loft)

    points = [(float(i), float(j), 0.3 * float((i * 7 + j * 3) % 5)) for i in range(5) for j in range(4)]
    own    = cmds.surface(degreeU=2, degreeV=1, knotU=[5, 5, 5.5, 7, 8, 8], knotV=[0, 0.4, 1.5, 2.0], point=points)
    add("own_knots_degree2_by_degree1", own)

    torus = cmds.torus(sections=8, spans=4, constructionHistory=False)[0]
    cmds.insertKnotSurface(shape_of(torus) + ".u[0.6]", constructionHistory=False, replaceOriginal=True)
    cmds.insertKnotSurface(shape_of(torus) + ".v[2.3]", constructionHistory=False, replaceOriginal=True)
    add("torus_inserted_knots", torus)
    return surfaces


def formatted(fixture):
    """the fixture as json: two-space nesting, each value of a surface on one line"""
    lines = ["{"]
    for key in fixture:
        if key != "surfaces":
            lines.append(f"  {json.dumps(key)}: {json.dumps(fixture[key])},")
    lines.append('  "surfaces": {')
    names = list(fixture["surfaces"])
    for i, name in enumerate(names):
        surface = fixture["surfaces"][name]
        keys    = list(surface)
        lines.append(f"    {json.dumps(name)}: {{")
        for j, key in enumerate(keys):
            lines.append(f"      {json.dumps(key)}: {json.dumps(surface[key])}" + ("," if j < len(keys) - 1 else ""))
        lines.append("    }" + ("," if i < len(names) - 1 else ""))
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else OUT)
    cmds.file(new=True, force=True)

    fixture = {
        "maya":     cmds.about(version=True),
        "space":    "object",
        "source":   SOURCE,
        "surfaces": record(),
    }
    text = formatted(fixture)
    assert json.loads(text) == fixture
    with open(out, "w", newline="\n") as f:
        f.write(text)
    print(f"{out}: {os.path.getsize(out)} bytes")
    for name, surface in fixture["surfaces"].items():
        print(
            f"    {name:30} degree {surface['degree_u']} x {surface['degree_v']}, "
            f"periodic {surface['periodic_u']} x {surface['periodic_v']}, "
            f"{len(surface['points'])} x {len(surface['points'][0])} CVs"
        )


if __name__ == "__main__":
    main()
    maya.standalone.uninitialize()
