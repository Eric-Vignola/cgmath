"""Array-shaping helpers for the transforms kernels, plus the package test runner.

:func:`_set_dimension` and :func:`_match_depth` are the two helpers every
public function uses to coerce its inputs; both are pure NumPy.
:func:`run_tests` runs the unit-test suite under ``_tests`` with a progress
counter and an encoding-safe output stream, and ``python -m cgmath.transforms.utils``
does the same from a shell.
"""

from __future__ import annotations

import functools
import locale
import os
import sys
import unittest
from typing import Any, Sequence

import numpy as np


def _set_dimension(
    data:           Any,
    ndim:           int        = 1,
    dtype:          np.ndarray = np.float64,
    reshape_matrix: bool       = False,
) -> np.ndarray:
    """Sets input data to expected dimension"""
    data = np.asarray(data, dtype=dtype)

    while data.ndim < ndim:
        data = data[np.newaxis]

    # For when matrices are given as lists of 16 floats
    if reshape_matrix:
        if data.shape[-1] == 16:
            data = data.reshape(-1, 4, 4)

    return data


def _match_depth(*data) -> list:
    """Levels given data to a common length, NumPy broadcasting rules.

    It is assumed all entries are already numpy arrays whose first axis is
    the batch axis. An input is accepted when its length equals the longest
    length, or is exactly 1; a length-1 input is expanded with
    :func:`numpy.broadcast_to`, which is a zero-copy read-only view.

    Any other length raises. Padding a short input -- by repeating its last
    row, recycling it, or truncating the long one -- turns an off-by-one in
    the caller into plausible-looking numbers instead of an error, so it is
    not done here. This matches :mod:`numpy` and
    :class:`scipy.spatial.transform.Rotation`.
    """
    counts  = [len(d) for d in data]
    highest = max(counts)

    if any(c not in (1, highest) for c in counts):
        raise ValueError(
            f"length mismatch: got {counts}; every input must be "
            f"length {highest} or 1"
        )

    return [
        d if len(d) == highest else np.broadcast_to(d, (highest,) + d.shape[1:])
        for d in data
    ]


class _ProgressResult(unittest.TextTestResult):
    """TextTestResult that prefixes each verbose test line with ``[N/total]``.

    ``startTest`` writes the counter and then defers to the stdlib for the
    description and `` ... ``, so the line format stays whatever this Python's
    unittest produces. The hook has the same shape on 3.7 (Maya 2022) and 3.11
    (Maya 2025). ``**kwargs`` absorbs the ``durations=`` that 3.12+ passes.
    Only active when ``showAll`` is set, i.e. verbosity 2 -- dot mode is untouched.
    """

    def __init__(self, stream, descriptions, verbosity, total=0, **kwargs):
        super().__init__(stream, descriptions, verbosity, **kwargs)
        self._total = total
        self._width = len(str(total))

    def startTest(self, test):
        if self.showAll:
            self.stream.write(
                "[%*d/%d] " % (self._width, self.testsRun + 1, self._total)
            )
        super().startTest(test)


def _run(suite, stream, verbosity, failfast):
    """Run ``suite`` with the progress-counting result class."""
    return unittest.TextTestRunner(
        stream    = stream,
        verbosity = verbosity,
        failfast  = failfast,
        resultclass=functools.partial(
            _ProgressResult, total=suite.countTestCases()
        ),
    ).run(suite)


def _encoding_safe(stream):
    """Wrap ``stream`` so nothing written through it can raise UnicodeEncodeError.

    Text is escaped against the platform default codec -- the one a downstream
    ``open()`` or ``logging.FileHandler`` uses when given no encoding -- before
    it reaches the real stream. Everything the codec can carry passes through
    untouched; anything it cannot is written as its escaped code point. On Windows that
    codec is cp1252, so em-dashes and degree signs still render and only true
    exotics are escaped.

    Delegation keeps whatever ``sys.stderr`` really is -- Maya's Script Editor,
    a studio log tee -- in the loop, so those still receive every line, now
    guaranteed encodable. Without this, one ``->`` written as U+2192 in a test
    docstring can turn into a recursive logging flood when the tee's handler
    cannot encode it and reports the failure back through the same stream.
    """
    codec = locale.getpreferredencoding(False) or "ascii"

    class _Safe:
        def write(self, text):
            return stream.write(text.encode(codec, "backslashreplace").decode(codec))

        def flush(self):
            return stream.flush()

        def __getattr__(self, name):
            return getattr(stream, name)

    return _Safe()


def run_tests(
    target:    str | Sequence[str] | None = None,
    verbosity: int                        = 2,
    failfast:  bool                       = False,
) -> unittest.TestResult:
    """
    runs the package's unit test suite

    Args:
        target: what to run, coarsest to finest::

            None                    the whole suite
            "test_*.py"                             a filename glob
            "test_matrix"                           one module
            "test_matrix.TestMatrix"                one class
            "test_matrix.TestMatrix.testToEuler"    one test method

            A list or tuple runs several in one pass, which is the quick way
            to re-run a handful of failures.

            Short dotted names are resolved inside the package, so you never
            write the fully qualified ``cgmath.transforms._tests....`` path.
            Pasting the long form copied out of a failure line works too.
        verbosity: 0 for silent, 1 for a dot per test, 2 for a line each.
        failfast: stop on the first failure or error.

    Returns:
        The :class:`unittest.TestResult`.  ``result.wasSuccessful()`` is the
        pass/fail answer; ``result.errors`` and ``result.failures`` carry the
        detail.

    Raises:
        ValueError: a dotted target naming no such module, class or method.
            Unittest would otherwise fold that into the run as an ordinary
            test error, which reads like a real failure rather than a typo.

    Note:
        Discovery imports every matching module, so a missing optional
        dependency shows up as an error against that module rather than
        aborting the run.

        >>> from cgmath.transforms.utils import run_tests
        >>> run_tests()                        # everything
        >>> run_tests("test_*.py")                            # a glob
        >>> run_tests("test_matrix.TestMatrix")               # one class
        >>> run_tests("test_matrix.TestMatrix.testToEuler")   # one test
        >>> run_tests(["test_vector", "test_euler"])          # two modules
        >>> run_tests(verbosity=1, failfast=True)
    """
    package_root = os.path.dirname(os.path.abspath(__file__))
    start_dir    = os.path.join(package_root, "_tests")

    if not os.path.isdir(start_dir):
        raise FileNotFoundError(f"no test directory at {start_dir!r}")

    # Modules import under the package's real dotted name --
    # ``cgmath.transforms._tests.<name>`` here, ``transforms._tests.<name>`` in
    # the standalone repository -- rather than as a bare ``<name>``, so
    # top_level_dir is the directory ABOVE the top-level package.  The suite
    # imports its own siblings by that dotted path, so pointing this at
    # ``_tests`` would break them.
    spec_name     = __spec__.name if __spec__ is not None else __name__
    package       = spec_name.rpartition(".")[0] or os.path.basename(package_root)
    top_level_dir = package_root
    for _ in package.split("."):
        top_level_dir = os.path.dirname(top_level_dir)
    root   = f"{package}._tests."
    loader = unittest.TestLoader()

    stream = _encoding_safe(sys.stderr)   # resolved now, so an installed stderr tee is seen

    if target is None:
        targets: list[str] = []
    elif isinstance(target, str):
        targets = [target]
    else:
        targets = list(target)

    if not targets:
        suite = loader.discover(
            start_dir, pattern="test_*.py", top_level_dir=top_level_dir
        )
        return _run(suite, stream, verbosity, failfast)

    # Namespaces a short name may live in: ``_tests`` itself, plus any test
    # subpackage inside it.  Callers should not have to know whether a module
    # sits at the top level or one directory down.
    namespaces = [root] + [
        root + entry + "."
        for entry in sorted(os.listdir(start_dir))
        if os.path.isfile(os.path.join(start_dir, entry, "__init__.py"))
    ]

    suite = unittest.TestSuite()
    for item in targets:
        if "*" in item or "?" in item or item.endswith(".py"):
            suite.addTests(
                loader.discover(start_dir, pattern=item, top_level_dir=top_level_dir)
            )
            continue

        # accept the short "test_x...", "_tests.test_x..." and the fully
        # qualified "cgmath.transforms._tests.test_x..." that a failure line prints
        name = item[len("_tests.") :] if item.startswith("_tests.") else item
        candidates = (
            [name] if name.startswith(root) else [ns + name for ns in namespaces]
        )

        resolved = None
        for candidate in candidates:
            mark  = len(loader.errors)
            found = loader.loadTestsFromName(candidate)
            if len(loader.errors) == mark:
                resolved = found
                break
            del loader.errors[mark:]  # discard the miss, try the next namespace

        if resolved is None:
            raise ValueError(
                "could not resolve test target "
                + repr(item)
                + "; tried "
                + ", ".join(candidates)
            )
        suite.addTests(resolved)

    return _run(suite, stream, verbosity, failfast)


# ``python -m cgmath.transforms.utils`` runs the suite and exits non-zero on failure, which
# is what a CI step or a pre-commit hook wants.  An optional argument narrows
# the run: ``python -m cgmath.transforms.utils test_matrix.TestMatrix``
if __name__ == "__main__":
    sys.exit(0 if run_tests(*sys.argv[1:2]).wasSuccessful() else 1)
