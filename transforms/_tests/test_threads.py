"""Small inputs run on one thread, with the answers they get on every thread.

Every kernel is ``parallel=True``, and a parallel call pays to start the
thread pool whatever its size, so a loop of calls on one matrix at a time
spent its time waking threads. :func:`cgmath.transforms._numba.sized`
hands the public functions each kernel's single-threaded twin below
``SINGLE_THREAD_BELOW`` elements. The twin must answer what the kernel
answers bit for bit. Under fast math they did not: six kernels -- the
quaternion product and slerp, the vector normalise, slerp and angle, and
the arc quaternion -- disagreed in the last bits on 3 to 36% of the values
of the 14 public functions built on them.
"""

import json
import os
import subprocess
import sys
import unittest
from contextlib import ExitStack, contextmanager
from unittest import mock

import numpy as np
from numba.core.registry import CPUDispatcher
from numba.core.runtime import _nrt_python, rtsys

from cgmath import transforms
from cgmath.transforms import _numba, main
from cgmath.transforms._numba import _axis, _euler, _matrix, _quaternion, _vector

THRESHOLD = _numba.SINGLE_THREAD_BELOW
MODULES   = (_axis, _euler, _matrix, _quaternion, _vector)

# public functions that call no kernel
PURE = {
    "matrix_flatten",
    "matrix_decompose",
    "matrix_weighted_rotational",
    "matrix_weighted_transformation",
    "euler_random",
}


def _unit(rng, n, k):
    a = rng.normal(size=(n, k))
    return a / np.linalg.norm(a, axis=1)[:, None]


def _quat(rng, n):
    return _unit(rng, n, 4)


def _vec(rng, n):
    return rng.normal(size=(n, 3))


def _mat(rng, n):
    m           = rng.normal(size=(n, 4, 4))
    m[:, :3, 3] = 0.0
    m[:, 3, 3]  = 1.0
    return m


def _axes(rng, n):
    return rng.integers(0, 6, n)


def _axis3(rng, n):
    return rng.integers(0, 3, n)


def _curves(rng, n):
    """euler_filter's arguments for n euler angles: a random walk in 2
    curves or more where n allows, since one curve runs on one thread"""
    count = next((k for k in range(2, n // 2 + 1) if n % k == 0), 1)
    euler = np.cumsum(_vec(rng, n).reshape(n // count, count, 3), axis=0)
    return euler, _axes(rng, count)


# random inputs of n elements for every public function that calls a kernel
CASES = {
    "quaternion_slerp": lambda r, n: (_quat(r, n), _quat(r, n), r.random(n)),
    "quaternion_dot": lambda r, n: (_quat(r, n), _quat(r, n)),
    "quaternion_conjugate": lambda r, n: (r.normal(size=(n, 4)),),
    "quaternion_inverse": lambda r, n: (r.normal(size=(n, 4)),),
    "quaternion_negate": lambda r, n: (r.normal(size=(n, 4)),),
    "quaternion_multiply": lambda r, n: (r.normal(size=(n, 4)), r.normal(size=(n, 4))),
    "quaternion_add": lambda r, n: (r.normal(size=(n, 4)), r.normal(size=(n, 4))),
    "quaternion_sub": lambda r, n: (r.normal(size=(n, 4)), r.normal(size=(n, 4))),
    "quaternion_to_matrix": lambda r, n: (_quat(r, n),),
    "quaternion_normalize": lambda r, n: (r.normal(size=(n, 4)),),
    "quaternion_to_euler": lambda r, n: (_quat(r, n), _axes(r, n)),
    "quaternion_random": lambda r, n: (n, 7),
    "quaternion_log": lambda r, n: (_quat(r, n),),
    "quaternion_exp": lambda r, n: (_vec(r, n),),
    "quaternion_nlerp": lambda r, n: (_quat(r, n), _quat(r, n), r.random(n)),
    "quaternion_intermediate": lambda r, n: (_quat(r, n), _quat(r, n), _quat(r, n)),
    "quaternion_squad": lambda r, n: (
        _quat(r, n), _quat(r, n), _quat(r, n), _quat(r, n), r.random(n)
    ),
    "matrix_identity": lambda r, n: (n,),
    "matrix_to_euler": lambda r, n: (_mat(r, n), _axes(r, n)),
    "matrix_to_quaternion": lambda r, n: (_mat(r, n),),
    "matrix_normalize": lambda r, n: (_mat(r, n),),
    "matrix_inverse": lambda r, n: (_mat(r, n),),
    "matrix_transpose": lambda r, n: (_mat(r, n),),
    "matrix_point_multiply": lambda r, n: (_vec(r, n), _mat(r, n)),
    "matrix_multiply": lambda r, n: (_mat(r, n), _mat(r, n)),
    "matrix_slerp": lambda r, n: (_mat(r, n), _mat(r, n), r.random(n)),
    "matrix_interpolate": lambda r, n: (_mat(r, n), _mat(r, n), r.random(n)),
    "matrix_local": lambda r, n: (_mat(r, n), _mat(r, n)),
    "matrix_random": lambda r, n: (n, 7, True),
    "matrix_delta": lambda r, n: (_mat(r, n), _mat(r, n)),
    "euler_to_matrix": lambda r, n: (_vec(r, n) * 3, _axes(r, n)),
    "euler_to_quaternion": lambda r, n: (_vec(r, n) * 3, _axes(r, n)),
    "euler_slerp": lambda r, n: (
        _vec(r, n) * 3, _vec(r, n) * 3, r.random(n), _axes(r, n), _axes(r, n), _axes(r, n)
    ),
    "euler_reorder": lambda r, n: (_vec(r, n) * 3, _axes(r, n), _axes(r, n)),
    "euler_filter": _curves,
    "axis_angle_to_quaternion": lambda r, n: (_vec(r, n), r.normal(size=n) * 3),
    "axis_angle_to_matrix": lambda r, n: (_vec(r, n), r.normal(size=n) * 3),
    "axis_angle_to_euler": lambda r, n: (_vec(r, n), r.normal(size=n) * 3, _axes(r, n)),
    "vector_to_matrix": lambda r, n: (_vec(r, n), _vec(r, n), _axis3(r, n), _axis3(r, n)),
    "vector_to_quaternion": lambda r, n: (_vec(r, n), _vec(r, n), _axis3(r, n), _axis3(r, n)),
    "vector_to_euler": lambda r, n: (
        _vec(r, n), _vec(r, n), _axis3(r, n), _axis3(r, n), _axes(r, n)
    ),
    "vector_cross": lambda r, n: (_vec(r, n), _vec(r, n)),
    "vector_dot": lambda r, n: (_vec(r, n), _vec(r, n)),
    "vector_magnitude": lambda r, n: (_vec(r, n),),
    "vector_normalize": lambda r, n: (_vec(r, n),),
    "vector_slerp": lambda r, n: (_vec(r, n), _vec(r, n), r.random(n)),
    "vector_lerp": lambda r, n: (_vec(r, n), _vec(r, n), r.random(n)),
    "vector_arc_to_quaternion": lambda r, n: (_vec(r, n), _vec(r, n)),
    "vector_arc_to_matrix": lambda r, n: (_vec(r, n), _vec(r, n)),
    "vector_arc_to_euler": lambda r, n: (_vec(r, n), _vec(r, n), _axes(r, n)),
    "vector_angle": lambda r, n: (_vec(r, n), _vec(r, n)),
    "vector_random": lambda r, n: (n, 7, True),
}


def _kernels():
    return [
        obj
        for module in MODULES
        for obj in vars(module).values()
        if isinstance(obj, CPUDispatcher) and obj.targetoptions.get("parallel")
    ]


def _run(name, size, threshold):
    """``name`` on ``size`` random elements, kernels picked against ``threshold``"""
    args = CASES[name](np.random.default_rng(sum(map(ord, name)) + size), size)
    with mock.patch.object(_numba, "SINGLE_THREAD_BELOW", threshold):
        return np.asarray(getattr(transforms, name)(*args))


class _Threaded:
    """Stands in for a threaded kernel: sized() hands out its twin, running
    it fails"""

    def __init__(self, kernel):
        self.py_func       = kernel.py_func
        self.targetoptions = kernel.targetoptions

    def __call__(self, *args):
        raise AssertionError("%s ran on every thread" % self.py_func.__name__)


class _Twin:
    """Stands in for a kernel's twin: running it fails"""

    def __init__(self, kernel):
        self.name = kernel.py_func.__name__

    def __call__(self, *args):
        raise AssertionError("%s ran on one thread" % self.name)


@contextmanager
def _no_threaded_kernels():
    """every threaded kernel replaced by a stand-in that fails when run,
    while sized() hands out the real twins"""
    twins = {}
    with ExitStack() as stack:
        for module in MODULES:
            for name, obj in list(vars(module).items()):
                if isinstance(obj, CPUDispatcher) and obj.targetoptions.get("parallel"):
                    stand_in        = _Threaded(obj)
                    twins[stand_in] = _numba.sized(obj, 0)
                    stack.enter_context(mock.patch.object(module, name, stand_in))
        stack.enter_context(mock.patch.dict(_numba._TWINS, twins))
        yield


def _one_curve(frames):
    return np.cumsum(np.random.default_rng(frames).normal(size=(frames, 3)), axis=0)


class TestDispatch(unittest.TestCase):
    def test_below_the_threshold_a_kernel_runs_its_twin(self):
        for kernel in _kernels():
            name = kernel.py_func.__name__
            twin = _numba.sized(kernel, THRESHOLD - 1)
            self.assertIsNot(twin, kernel, name)
            self.assertIs(_numba.sized(kernel, 0),             twin,   name)
            self.assertIs(_numba.sized(kernel, THRESHOLD),     kernel, name)
            self.assertIs(_numba.sized(kernel, THRESHOLD + 1), kernel, name)
            self.assertFalse(twin.targetoptions.get("parallel"), name)

    def test_the_twin_is_the_kernels_body_without_threads(self):
        for kernel in _kernels():
            twin = _numba.sized(kernel, 1)
            name = kernel.py_func.__name__
            self.assertIs(twin.py_func.__code__, kernel.py_func.__code__, name)
            self.assertEqual(twin.targetoptions.get("error_model"), "numpy", name)
            options = dict(kernel.targetoptions)
            options.pop("parallel")
            options["error_model"] = "numpy"
            self.assertEqual(twin.targetoptions, options, name)

    def test_the_twin_is_cached_apart_from_its_kernel(self):
        # numba keys a cache file on module, qualified name and first line:
        # the same function compiled twice would overwrite one cache with
        # the other, and a process could load the threaded kernel as the twin
        for kernel in _kernels():
            twin = _numba.sized(kernel, 1)
            name = kernel.py_func.__name__
            self.assertIsNotNone(twin._cache, name)
            self.assertNotEqual(
                twin._cache._cache_file._index_name,
                kernel._cache._cache_file._index_name,
                name,
            )

    def test_no_kernel_uses_fast_math(self):
        found = [
            obj.py_func.__name__
            for module in MODULES
            for obj in vars(module).values()
            if isinstance(obj, CPUDispatcher) and obj.targetoptions.get("fastmath")
        ]
        self.assertEqual(found, [])

    def test_a_division_by_zero_on_one_thread_is_a_nan_as_on_every_thread(self):
        zero = np.zeros((1, 3))
        for threshold in (0, sys.maxsize):
            with mock.patch.object(_numba, "SINGLE_THREAD_BELOW", threshold):
                self.assertTrue(np.isnan(transforms.vector_normalize(zero)).all())


class TestPublicFunctions(unittest.TestCase):
    """every public function fetches its kernels through sized(): one that
    called a kernel itself would wake every thread for one element again"""

    def test_a_small_call_never_runs_a_threaded_kernel(self):
        with _no_threaded_kernels():
            for name in CASES:
                for size in (1, THRESHOLD - 1):
                    with self.subTest(name=name, size=size):
                        _run(name, size, THRESHOLD)

    def test_one_curve_never_runs_a_threaded_kernel(self):
        # euler_filter's threads share out curves, not frames, which run in
        # order: one curve of 1000 frames took 17 ms on 24 threads with every
        # core busy, where one thread takes 0.03 ms
        with _no_threaded_kernels():
            for frames in (THRESHOLD, THRESHOLD + 1, 50 * THRESHOLD):
                for shape in ((frames, 3), (frames, 1, 3)):
                    euler = _one_curve(frames).reshape(shape)
                    with self.subTest(shape=shape):
                        transforms.euler_filter(euler, 4)

    def test_a_large_call_never_runs_a_twin(self):
        twins = {kernel: _Twin(kernel) for kernel in _kernels()}
        with mock.patch.dict(_numba._TWINS, twins):
            for name in CASES:
                with self.subTest(name=name):
                    _run(name, THRESHOLD, THRESHOLD)

    def test_one_thread_allocates_no_more_than_every_thread(self):
        # a scratch array made in a kernel's loop is hoisted out of the
        # threaded loop but made once an element in a plain one: the arc
        # quaternion's twin made 3 an element and took 2.4 times as long
        enabled = _nrt_python.memsys_stats_enabled()
        _nrt_python.memsys_enable_stats()
        try:
            for name in CASES:
                counts = []
                for threshold in (0, sys.maxsize):
                    _run(name, 100, threshold)
                    before = rtsys.get_allocation_stats().alloc
                    _run(name, 100, threshold)
                    counts.append(rtsys.get_allocation_stats().alloc - before)
                threaded, one = counts
                with self.subTest(name=name):
                    self.assertLessEqual(one, threaded)
        finally:
            if not enabled:
                _nrt_python.memsys_disable_stats()

    def test_a_small_call_runs_with_the_jit_off(self):
        # NUMBA_DISABLE_JIT=1 (a debugger, a coverage run) leaves every
        # kernel a python function, with no dispatcher to copy a twin from.
        # the child sees what this process sees: cgmath, and a runner's venv
        env = dict(
            os.environ,
            NUMBA_DISABLE_JIT = "1",
            PYTHONPATH        = os.pathsep.join(x for x in sys.path if x),
        )
        out = subprocess.run(
            [sys.executable, "-c", _JIT_OFF_SCRIPT],
            capture_output = True,
            text           = True,
            env            = env,
            timeout        = 300,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        matrix, angle = (json.loads(line) for line in out.stdout.splitlines()[-2:])
        np.testing.assert_allclose(matrix, transforms.euler_to_matrix((0.3, 0.2, 0.1)))
        np.testing.assert_allclose(angle, [np.pi / 2])


_JIT_OFF_SCRIPT = """
import json

from cgmath import transforms as T

m = T.euler_to_matrix((0.3, 0.2, 0.1))
print(json.dumps(T.matrix_multiply(m, T.matrix_identity(1)).tolist()))
print(json.dumps(T.vector_angle((1, 0, 0), (0, 1, 0)).tolist()))
"""


class TestSameAnswers(unittest.TestCase):
    def assertSameBits(self, got, want, where):
        """bit for bit, signed zeros included; a nan wherever ``want`` has one"""
        self.assertEqual(got.shape, want.shape, where)
        self.assertEqual(got.dtype, want.dtype, where)
        nan = np.isnan(want)
        np.testing.assert_array_equal(np.isnan(got), nan, err_msg=where)
        bits = got.dtype.str[1:].replace("f", "u")
        np.testing.assert_array_equal(
            got[~nan].view(bits), want[~nan].view(bits), err_msg=where
        )

    def test_every_public_function_is_covered(self):
        constants = {"XYZ", "YZX", "ZXY", "XZY", "YXZ", "ZYX", "X", "Y", "Z"}
        self.assertEqual(set(main.__all__) - constants - PURE, set(CASES))

    def test_every_public_function_on_both_sides_of_the_threshold(self):
        for name in CASES:
            for size in (THRESHOLD - 1, THRESHOLD, THRESHOLD + 1):
                where    = "%s on %d" % (name, size)
                threaded = _run(name, size, 0)
                one      = _run(name, size, sys.maxsize)
                picked   = _run(name, size, THRESHOLD)
                self.assertSameBits(one, threaded, where)
                self.assertSameBits(picked, threaded, where)

    def test_one_curve_on_both_sides_of_the_threshold(self):
        for frames in (THRESHOLD - 1, THRESHOLD, THRESHOLD + 1):
            euler = _one_curve(frames)
            got   = []
            for threshold in (0, sys.maxsize, THRESHOLD):
                with mock.patch.object(_numba, "SINGLE_THREAD_BELOW", threshold):
                    got.append(transforms.euler_filter(euler, 4))
            threaded, one, picked = got
            self.assertSameBits(one, threaded, "one curve of %d" % frames)
            self.assertSameBits(picked, threaded, "one curve of %d" % frames)

    def test_a_zero_vector_is_a_right_angle_from_anything(self):
        # the kernel read the vector last normalised in its scratch space,
        # which differs between one thread and many
        rng       = np.random.default_rng(3)
        size      = THRESHOLD + 7
        v0        = rng.normal(size=(size, 3))
        v1        = rng.normal(size=(size, 3))
        v0[::5]   = 0.0
        v1[::7]   = 0.0
        zero      = np.zeros(size, dtype=bool)
        zero[::5] = True
        zero[::7] = True
        for threshold in (0, sys.maxsize):
            with mock.patch.object(_numba, "SINGLE_THREAD_BELOW", threshold):
                angle = transforms.vector_angle(v0, v1)
            self.assertTrue(np.all(angle[zero] == np.pi / 2), threshold)
            self.assertTrue(np.all(np.isfinite(angle)), threshold)


if __name__ == "__main__":
    unittest.main()
