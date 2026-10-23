"""Every cgmath entry point imports first in a new interpreter.

cgmath.geometry imports cgmath.hierarchy at module scope (mesh.py,
skin_weights.py, deform/skin_deform.py), and cgmath.hierarchy imports
cgmath.formats.glb and, where the FBX SDK is, cgmath.formats.fbx. Neither of
those two imports back at module scope: glb takes MeshData when it reads
meshes, fbx takes the hierarchy helpers inside the functions that use them.
An import back is a cycle, and which module loads first decides how it shows:
an ImportError from a partially initialized module for one entry point, and
for another a module that imports but with its FBX handles None, as
hierarchy's import of the FBX side sits in a try / except ImportError.
"""

import importlib.util
import json
import unittest
from concurrent.futures import ThreadPoolExecutor

from cgmath._tests.test_data_module import _fresh

# where an import can start: a submodule runs its package's __init__ first,
# so one entry per package is every order the cycle could load in
ENTRIES = (
    "cgmath.data",
    "cgmath.transforms",
    "cgmath.constraints",
    "cgmath.rbf",
    "cgmath.formats.glb",
    "cgmath.formats.fbx",
    "cgmath.hierarchy",
    "cgmath.geometry",
    "cgmath.render",
)

# after the entry, the rest in one order, so every run loads every entry and
# each module's names are compared; then, for every cgmath module loaded, its
# module-level names bound to None
FINGERPRINT = """
import importlib, json, sys

importlib.import_module({entry!r})
for name in {entries!r}:
    importlib.import_module(name)
print(json.dumps({{
    name: sorted(k for k, v in vars(module).items() if v is None and not k.startswith("__"))
    for name, module in sorted(sys.modules.items())
    if name == "cgmath" or name.startswith("cgmath.")
}}))
"""

HAS_FBX = importlib.util.find_spec("fbx") is not None


class TestImportOrder(unittest.TestCase):
    def test_hierarchy_loads_nothing_of_geometry(self):
        # scipy comes with trimesh, which formats.glb imports where it is installed
        out = _fresh(
            """
            import sys

            before = set(sys.modules)
            import cgmath.hierarchy
            added = sorted(set(sys.modules) - before)
            print([x for x in added if x.startswith("cgmath.geometry")])
            print([x for x in added if x.split(".")[0] == "numba"])
            """
        )
        self.assertEqual(out.splitlines()[-2:], ["[]", "[]"])

    def test_every_entry_point_imports_first(self):
        codes = [FINGERPRINT.format(entry=entry, entries=ENTRIES) for entry in ENTRIES]
        with ThreadPoolExecutor(len(ENTRIES)) as pool:
            outs = list(pool.map(_fresh, codes))
        prints = {entry: json.loads(out.splitlines()[-1]) for entry, out in zip(ENTRIES, outs)}

        # every run loads every entry, so whichever module came first, the
        # same modules load and bind the same names to None: whole
        # fingerprints are compared
        first = prints[ENTRIES[0]]
        for entry in ENTRIES[1:]:
            with self.subTest(entry=entry):
                self.assertEqual(prints[entry], first)
        # and the FBX side is there wherever the SDK is
        for name in ("FBX", "FbxExporter", "_ascii_read_back", "_user_property_value"):
            self.assertEqual(name in first["cgmath.hierarchy.hierarchy"], not HAS_FBX, name)
        self.assertEqual("fbx" in first["cgmath.geometry.mesh"], not HAS_FBX)

    @unittest.skipIf(
        importlib.util.find_spec("trimesh") is None or importlib.util.find_spec("pygltflib") is None,
        "GlbData.mesh_list needs trimesh and pygltflib",
    )
    def test_glb_loads_geometry_when_it_reads_meshes(self):
        out = _fresh(
            """
            import os, sys, tempfile

            import trimesh
            from cgmath.formats.glb import GlbData

            print("geometry", "cgmath.geometry" in sys.modules)
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "box.glb")
                trimesh.creation.box().export(path)
                meshes = GlbData(path).mesh_list
            print(type(meshes).__name__, [type(x).__name__ for x in meshes], [len(x.points) for x in meshes])
            print("geometry", "cgmath.geometry" in sys.modules)
            """
        )
        self.assertEqual(
            out.splitlines()[-3:],
            ["geometry False", "MeshList ['MeshData'] [8]", "geometry True"],
        )


if __name__ == "__main__":
    unittest.main()
