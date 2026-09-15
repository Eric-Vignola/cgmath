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
import re
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
        """Every test package must be importable by dotted name.

        ``cgmath/_tests/__init__.py`` makes that suite a real package, and
        ``cgmath/transforms/_tests/__init__.py`` does the same for the transforms
        suite, which means sibling test helpers have to be imported as
        ``cgmath._tests.<name>`` / ``cgmath.transforms._tests.<name>`` rather than
        bare ``<name>``. A bare sibling import passes when the tests are run from
        inside the directory and fails everywhere else.

        The test packages are found by walking the library, the same way
        :func:`cgmath.utils.run_tests` finds them.
        """
        names = []
        for dirpath, dirnames, filenames in os.walk(PACKAGE_ROOT):
            dirnames[:] = sorted(
                d for d in dirnames if d != "__pycache__" and not d.startswith(".")
            )
            if os.path.basename(dirpath) == "_tests" and "__init__.py" in filenames:
                dotted = ".".join(
                    ["cgmath"] + os.path.relpath(dirpath, PACKAGE_ROOT).split(os.sep)
                )
                names += [
                    f"{dotted}.{f[:-3]}"
                    for f in sorted(filenames)
                    if f.startswith("test_") and f.endswith(".py")
                ]
                dirnames[:] = []
        self.assertGreater(len(names), 10)
        self.assertTrue(
            any(n.startswith("cgmath.transforms._tests.") for n in names),
            "the walk found no cgmath.transforms._tests modules",
        )

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
    """``transforms`` ships inside cgmath as :mod:`cgmath.transforms`, and the
    hierarchy types are not part of it.  Both halves of that split are easy to
    undo by accident.
    """

    # A statement that imports the TOP-LEVEL package: ``import transforms``,
    # ``import transforms as tr``, ``from transforms import x``,
    # ``from transforms.main import x``.  ``from cgmath.transforms ...`` does not
    # match, and neither does prose that merely mentions the name.
    _TOP_LEVEL = re.compile(r"^[ \t]*(?:import[ \t]+transforms\b|from[ \t]+transforms[ \t.])", re.M)

    def test_transforms_is_a_cgmath_subpackage(self):
        """``cgmath.transforms`` must resolve from inside the package."""
        from cgmath import transforms

        self.assertTrue(
            os.path.abspath(transforms.__file__).startswith(PACKAGE_ROOT),
            "cgmath.transforms resolved from outside cgmath",
        )

    def test_nothing_imports_the_standalone_transforms(self):
        """No module may import the top-level ``transforms`` package.

        The standalone repository installs under that name too, so a stray
        top-level import passes wherever that copy happens to be installed and
        fails everywhere else.
        """
        offenders = []
        for dirpath, dirnames, filenames in os.walk(PACKAGE_ROOT):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8") as fh:
                    if self._TOP_LEVEL.search(fh.read()):
                        offenders.append(os.path.relpath(path, PACKAGE_ROOT))
        self.assertEqual([], offenders)

    def test_hierarchy_types_come_from_facet(self):
        """The hierarchy types live in ``cgmath.hierarchy``, not ``cgmath.transforms``."""
        from cgmath import transforms

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
                f"cgmath.transforms exports {cls.__name__}",
            )


if __name__ == "__main__":
    unittest.main()
