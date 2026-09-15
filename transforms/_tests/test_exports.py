"""The package root and ``cgmath.transforms.main.__all__`` must export the same names.

Five quaternion functions once sat in ``main.__all__`` without being
re-exported from the root, and ``euler_filter`` was re-exported without being
in ``__all__``. Both slipped through because nothing compared the two lists.
These tests do, in both directions.
"""

import inspect
import unittest

from cgmath import transforms
from cgmath.transforms import main


def _public_functions(module):
    return {
        name
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and inspect.isfunction(obj)
        and obj.__module__ == "cgmath.transforms.main"
    }


class TestRootExports(unittest.TestCase):
    def test_every_name_in_main_all_is_at_the_root(self):
        self.assertEqual([], [n for n in main.__all__ if not hasattr(transforms, n)])

    def test_root_and_main_refer_to_the_same_objects(self):
        self.assertEqual(
            [], [n for n in main.__all__ if getattr(transforms, n) is not getattr(main, n)]
        )

    def test_every_public_function_at_the_root_is_in_main_all(self):
        self.assertEqual(set(), _public_functions(transforms) - set(main.__all__))

    def test_every_public_function_in_main_is_in_main_all(self):
        self.assertEqual(set(), _public_functions(main) - set(main.__all__))


if __name__ == "__main__":
    unittest.main()
