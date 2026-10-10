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

import ast
import collections
import importlib
import importlib.util
import os
import pkgutil
import re
import tempfile
import types
import unittest

import cgmath

# Imports guarded by ``try: ... except ImportError`` inside the package are
# fine -- they degrade to ``None``. This set is for modules that cannot import
# at all without a dependency we do not require.
OPTIONAL = {
    "cgmath.usd",  # external studio package, not yet vendored
    "cgmath.formats.fbx",  # hard-requires the Autodesk FBX SDK, which is not on PyPI
}
# `cgmath.formats.usd` imports pxr unguarded. pxr comes with the `usd` extra
# or from the host DCC, so the check still runs wherever pxr is available.
if importlib.util.find_spec("pxr") is None:
    OPTIONAL.add("cgmath.formats.usd")

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


# What python 3.9's TestCase and IsolatedAsyncioTestCase lack: assertNoLogs
# came in 3.10, the enter*Context methods in 3.11, the rest after 3.13.
_NEWER_TESTCASE_METHODS = {
    "assertNoLogs",
    "enterContext",
    "enterClassContext",
    "enterAsyncContext",
    "assertHasAttr",
    "assertNotHasAttr",
    "assertIsSubclass",
    "assertNotIsSubclass",
    "assertStartsWith",
    "assertNotStartsWith",
    "assertEndsWith",
    "assertNotEndsWith",
}


def _python_files(root):
    """
    Every .py file under *root*, relative to it, with ``/`` separators. A
    folder whose name starts with a dot, or named build or dist, is not the
    package: a git worktree under ``.claude``, a ``.venv``, setuptools'
    build/lib copy.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d != "__pycache__" and not d.startswith(".") and d not in ("build", "dist")
        )
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")


def _bar_annotations(tree, hints=False):
    """
    The line of every ``X | Y`` annotation that runs when its module, class
    or function statement does: a parameter's, a return's, and a variable's
    at module or class level. A local variable's annotation never runs.

    With *hints*, the line of every one typing.get_type_hints runs, as Data
    does for its fields: a variable's at module or class level, quoted or
    not, since the future import only quotes it, or quoted inside, as
    ``Optional['int | None']``.
    """

    def bar(annotation, quoted=True):
        """
        Whether *annotation* holds an ``X | Y``. With *hints*, a string in a
        type's place is read as the annotation it quotes, as get_type_hints
        reads ``'int | None'`` and ``Optional['int | None']``; a call's
        argument, a Literal's value and an Annotated's metadata are no type.
        """
        if annotation is None:
            return False
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            return True
        text = annotation.value if isinstance(annotation, ast.Constant) else None
        if hints and quoted and isinstance(text, str):
            try:
                return bar(ast.parse(text, mode="eval").body)
            except SyntaxError:
                return False
        if isinstance(annotation, ast.Subscript):
            index = annotation.slice
            kind  = getattr(annotation.value, "attr", getattr(annotation.value, "id", None))
            items = index.elts if isinstance(index, ast.Tuple) else [index]
            typed = 0 if kind == "Literal" else 1 if kind == "Annotated" else len(items)
            return bar(annotation.value, False) or any(
                bar(item, quoted and i < typed) for i, item in enumerate(items)
            )
        quoted = quoted and not isinstance(annotation, ast.Call)
        return any(bar(child, quoted) for child in ast.iter_child_nodes(annotation))

    lines = []

    def visit(node, in_function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
                if not hints and arg is not None and bar(arg.annotation):
                    lines.append(arg.lineno)
            if not hints and bar(node.returns):
                lines.append(node.returns.lineno)
            in_function = True
        elif isinstance(node, ast.ClassDef):
            in_function = False
        elif isinstance(node, ast.AnnAssign) and not in_function and bar(node.annotation):
            lines.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.stmt, ast.excepthandler)):  # expressions hold no annotation
                visit(child, in_function)

    visit(tree, False)
    return sorted(lines)


# Builtin types: an ``|`` with one of them or None on a side is of types, not
# of ints, bools or arrays
_BUILTIN_TYPES = {
    "bool",
    "bytearray",
    "bytes",
    "complex",
    "dict",
    "float",
    "frozenset",
    "int",
    "list",
    "memoryview",
    "object",
    "range",
    "set",
    "slice",
    "str",
    "tuple",
    "type",
}

# sys as python 3.9 has it, for an ``if`` on sys.version_info
_VersionInfo = collections.namedtuple("version_info", "major minor micro releaselevel serial")
_SYS_3_9     = types.SimpleNamespace(version_info=_VersionInfo(3, 9, 0, "final", 0))


def _on_3_9(test):
    """
    What *test* gives on python 3.9 when it reads sys.version_info and
    nothing else, as ``sys.version_info >= (3, 10)`` does; None otherwise.
    """
    names = {getattr(n, "attr", getattr(n, "id", None)) for n in ast.walk(test)}
    if "version_info" not in names:
        return None
    space = {"__builtins__": {}, "sys": _SYS_3_9, "version_info": _SYS_3_9.version_info}
    try:
        return bool(eval(compile(ast.Expression(test), "<test>", "eval"), space))
    except Exception:  # noqa: BLE001 - it reads more than that
        return None


def _bar_of_types(node):
    """
    Whether *node* is an ``X | Y`` of types, as far as the source tells: an
    ``|`` in isinstance's or issubclass's class argument, or one with None or
    a builtin type on a side. One of two other names may be of ints or arrays.
    """
    if isinstance(node, ast.Call):
        name = getattr(node.func, "id", None)
        if name not in ("isinstance", "issubclass") or len(node.args) != 2:
            return False
        ops = [n.op for n in ast.walk(node.args[1]) if isinstance(n, ast.BinOp)]
        return any(isinstance(op, ast.BitOr) for op in ops)
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.BitOr)
        and any(
            (isinstance(side, ast.Constant) and side.value is None)
            or (isinstance(side, ast.Name) and side.id in _BUILTIN_TYPES)
            for side in (node.left, node.right)
        )
    )


def _what_3_9_runs_and_lacks(tree):
    """
    What python 3.9 runs in *tree* and fails on, as ``(line, name)``: an
    ``X | Y`` of types run as an expression, named ``|``, and a unittest
    method its TestCase lacks, named by itself. Annotations are
    _bar_annotations' to read, and of an ``if`` on sys.version_info only the
    branch 3.9 takes is read.
    """
    found = set()
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Attribute):
            if node.attr in _NEWER_TESTCASE_METHODS:
                found.add((node.lineno, node.attr))
        elif isinstance(node, (ast.BinOp, ast.Call)):
            if _bar_of_types(node):
                found.add((node.lineno, "|"))
        elif isinstance(node, (ast.If, ast.IfExp)):
            taken = _on_3_9(node.test)
            if taken is not None:
                branch = node.body if taken else node.orelse
                stack += branch if isinstance(branch, list) else [branch]
                continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.arg, ast.AnnAssign)):
            annotation = getattr(node, "returns", None) or getattr(node, "annotation", None)
            stack += [c for c in ast.iter_child_nodes(node) if c is not annotation]
        else:
            stack += ast.iter_child_nodes(node)
    return sorted(found)


class TestPython39(unittest.TestCase):
    """
    pyproject says python 3.9 and the suite runs on 3.11, which accepts what
    3.9 refuses. On 3.9 an annotation runs when its ``def`` does, unless the
    module starts with ``from __future__ import annotations``, and ``X | Y``
    of two types raises TypeError there: the module does not import. The
    future import only quotes it: typing.get_type_hints still runs a class's
    annotations, and on 3.9 a Data field annotated so loses its Array spec.
    ``X | Y`` of types run as an expression, as ``isinstance(x, list | tuple)``,
    raises the same TypeError when its line runs, and a test calling a
    unittest method 3.9 lacks fails there too. These checks read the source,
    so they fail on any python.
    """

    @classmethod
    def setUpClass(cls):
        cls.trees, cls.syntax = {}, []
        for rel in _python_files(PACKAGE_ROOT):
            with open(os.path.join(PACKAGE_ROOT, rel), encoding="utf-8") as fh:
                source = fh.read()
            try:
                cls.trees[rel] = ast.parse(source, rel, feature_version=(3, 9))
            except SyntaxError as err:
                cls.syntax.append(f"{rel}:{err.lineno}: {err.msg}")
        # one walk of every expression serves the two checks that need one
        cls.lacks = {rel: _what_3_9_runs_and_lacks(tree) for rel, tree in cls.trees.items()}

    def test_the_walk_reads_the_package_only(self):
        """a nested worktree, a venv or a build copy holds code that is not cgmath's"""
        with tempfile.TemporaryDirectory() as root:
            for rel in (
                "a.py",
                "sub/b.py",
                "sub/c.txt",
                "__pycache__/d.py",
                ".claude/worktrees/older/e.py",
                ".venv/Lib/site-packages/f.py",
                "build/lib/cgmath/g.py",
                "dist/h.py",
                "sub/.hidden/i.py",
            ):
                path = os.path.join(root, *rel.split("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "w").close()
            self.assertEqual(["a.py", "sub/b.py"], list(_python_files(root)))

    def test_every_file_parses_as_python_3_9(self):
        """no match statement, ``except*`` or other syntax 3.9 lacks, as far as ast.parse tells"""
        self.assertGreater(len(self.trees) + len(self.syntax), 100)
        self.assertEqual([], self.syntax)

    def test_no_bar_annotation_runs_without_the_future_import(self):
        """before 3.10, ``X | Y`` in a signature needs ``from __future__ import annotations``"""
        offenders = []
        for rel, tree in self.trees.items():
            future = any(
                isinstance(n, ast.ImportFrom)
                and n.module == "__future__"
                and any(a.name == "annotations" for a in n.names)
                for n in tree.body
            )
            if not future:
                offenders += [f"{rel}:{line}" for line in _bar_annotations(tree)]
        self.assertEqual([], offenders)

    def test_the_hint_check_sees_what_the_future_import_quoted(self):
        """get_type_hints runs a module's and a class's variables, quoted whole, in part or not"""
        tree = ast.parse(
            "from __future__ import annotations\n"
            "a: int | None = None\n"
            "class B:\n"
            "    c: int | None = None\n"
            "    d: 'int | None' = None\n"
            "    e: Optional[int] = None\n"
            "    if True:\n"
            "        f: int | None = None\n"
            "    def g(self, h: int | None) -> int | None:\n"
            "        i: int | None = None\n"
            "        class J:\n"
            "            k: int | None = None\n"
            "try:\n"
            "    pass\n"
            "except ImportError:\n"
            "    m: int | None = None\n"
            "class N:\n"
            "    o: Optional['int | None'] = None\n"
            "    p: Callable[['int | None'], int] = None\n"
            "    q: Annotated['int | None', 'x | y'] = None\n"
            "    r: Annotated[int, 'x | y'] = 0\n"
            "    s: Literal['x | y'] = 'x'\n"
            "    t: Optional[Array(int, 'M | K')] = None\n"
        )
        self.assertEqual([2, 4, 5, 8, 12, 16, 18, 19, 20], _bar_annotations(tree, hints=True))
        self.assertEqual([2, 4, 8, 9, 9, 12, 16], _bar_annotations(tree))

    def test_no_bar_annotation_get_type_hints_runs(self):
        """future import or not, a Data field annotated ``A | None`` loses its Array spec on 3.9"""
        offenders = [
            f"{rel}:{line}"
            for rel, tree in self.trees.items()
            for line in _bar_annotations(tree, hints=True)
        ]
        self.assertEqual([], offenders)

    def test_the_expression_check_reads_what_3_9_runs(self):
        """no annotation, nor the branch of an ``if`` on sys.version_info that 3.9 does not take"""
        tree = ast.parse(
            "import sys\n"
            "def f(a: int | None, *b: list | tuple) -> int | None:\n"
            "    c: int | None = a | b\n"
            "    if isinstance(a, A | B):\n"
            "        return issubclass(a, (A, B | C))\n"
            "    return x.y | None, self.assertNoLogs()\n"
            "def g(d=None | str, e=lambda f=int | None: f): pass\n"
            "if sys.version_info >= (3, 10):\n"
            "    h = int | None\n"
            "    self.enterContext(h)\n"
            "else:\n"
            "    h = Optional[str | None]\n"
            "if sys.version_info < (3, 10):\n"
            "    i: Optional[int] = [bytes | None]\n"
            "j = bytes | None if sys.version_info.minor >= 10 else None\n"
            "k = bytes | None if version_info >= (3, 10) or k else None\n"
        )
        found = _what_3_9_runs_and_lacks(tree)
        self.assertEqual([4, 5, 6, 7, 12, 14, 16], [line for line, name in found if name == "|"])
        self.assertEqual([(6, "assertNoLogs")], [f for f in found if f[1] != "|"])

    def test_no_bar_of_types_runs_as_an_expression(self):
        """``isinstance(x, list | tuple)`` or ``x | None`` is a TypeError on 3.9 when it runs"""
        offenders = [
            f"{rel}:{line}"
            for rel, found in self.lacks.items()
            for line, name in found
            if name == "|"
        ]
        self.assertEqual([], offenders)

    def test_no_unittest_method_3_9_lacks(self):
        """``self.assertNoLogs(...)`` is an AttributeError on 3.9, found only when the test runs"""
        offenders = [
            f"{rel}:{line}: {name}"
            for rel, found in self.lacks.items()
            for line, name in found
            if name != "|"
        ]
        self.assertEqual([], offenders)


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
