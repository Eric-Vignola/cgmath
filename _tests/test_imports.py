"""Every module in the package must import cleanly.

The rest of the suite only imports what it happens to exercise, so a module
that nothing covers can sit broken indefinitely -- and a package-level change
(renaming a namespace, adding an ``__init__.py``, moving a helper) can break
imports in files no test touches.

This module walks the package and imports every submodule. It is deliberately
cheap and has no fixtures: it is the check that fails first and points at the
file, rather than surfacing later as a confusing error inside an unrelated
test.
"""

import importlib
import os
import pkgutil
import unittest

import cgmath

# Imports guarded by ``try: ... except ImportError`` inside the package are
# fine -- they degrade to ``None``. This set is for modules that cannot import
# at all without a dependency we do not require.
OPTIONAL = {
    "cgmath.usd",  # external studio package, not yet vendored
    "cgmath.formats.fbx",  # hard-requires the Autodesk FBX SDK, which is not on PyPI
}

PACKAGE_ROOT = os.path.dirname(os.path.abspath(cgmath.__file__))


def _iter_modules():
    """Yield every importable module name under :mod:`cgmath`, tests excluded."""
    for info in pkgutil.walk_packages([PACKAGE_ROOT], prefix="cgmath."):
        name = info.name
        if name.startswith("cgmath._tests"):
            continue
        if any(name == o or name.startswith(o + ".") for o in OPTIONAL):
            continue
        yield name


class TestEveryModuleImports(unittest.TestCase):
    def test_every_module_imports(self):
        modules = sorted(_iter_modules())
        self.assertGreater(len(modules), 20, "module discovery found almost nothing")

        broken = []
        for name in modules:
            try:
                importlib.import_module(name)
            except Exception as err:  # noqa: BLE001 - we want to report them all
                broken.append(f"{name}: {type(err).__name__}: {err}")

        self.assertEqual(
            [], broken, "\n  " + "\n  ".join(broken) if broken else ""
        )

    def test_test_modules_import(self):
        """The test package itself must be importable by dotted name.

        ``cgmath/_tests/__init__.py`` makes this a real package, which means
        sibling test helpers have to be imported as ``cgmath._tests.<name>``
        rather than bare ``<name>``. A bare sibling import passes when the
        tests are run from inside the directory and fails everywhere else.
        """
        names = [
            f"cgmath._tests.{f[:-3]}"
            for f in sorted(os.listdir(os.path.join(PACKAGE_ROOT, "_tests")))
            if f.startswith("test_") and f.endswith(".py")
        ]
        self.assertGreater(len(names), 10)

        broken = []
        for name in names:
            try:
                importlib.import_module(name)
            except Exception as err:  # noqa: BLE001
                broken.append(f"{name}: {type(err).__name__}: {err}")

        self.assertEqual(
            [], broken, "\n  " + "\n  ".join(broken) if broken else ""
        )


class TestTransformsBoundary(unittest.TestCase):
    """``transforms`` is an external package, and the hierarchy types are
    not part of it.  Both halves of that split are easy to undo by accident.
    """

    def test_transforms_is_an_external_dependency(self):
        """``transforms`` must resolve outside the package, not inside it."""
        import transforms

        self.assertFalse(
            os.path.abspath(transforms.__file__).startswith(PACKAGE_ROOT),
            "transforms resolved from inside cgmath; it is meant to be external",
        )

    def test_hierarchy_types_come_from_facet(self):
        """The hierarchy types moved out of ``transforms`` and into ``cgmath``."""
        import transforms

        from cgmath.hierarchy import (
            ClipData,
            HierarchyData,
            TransformData,
            TransformList,
        )

        for cls in (TransformData, TransformList, HierarchyData, ClipData):
            self.assertTrue(cls.__module__.startswith("cgmath."), cls.__module__)
            self.assertFalse(
                hasattr(transforms, cls.__name__),
                f"transforms still exports {cls.__name__}",
            )


if __name__ == "__main__":
    unittest.main()
