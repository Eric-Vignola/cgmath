"""Tests for :mod:`cgmath.formats.usd.prim`."""

import unittest

try:
    from pxr import Usd, UsdGeom
except ImportError:  # pragma: no cover - optional dependency
    Usd = None


@unittest.skipIf(Usd is None, "pxr (USD) is not installed")
class TestPrimInherits(unittest.TestCase):
    def setUp(self):
        from cgmath.formats.usd import prim as usd_prim
        from cgmath.formats.usd import stage as usd_stage

        self.usd_prim = usd_prim
        self.stage    = usd_stage.new_stage_in_memory(no_cache=True)
        self.body     = UsdGeom.Mesh.Define(self.stage, "/World/Body").GetPrim()
        self.rig      = UsdGeom.Xform.Define(self.stage, "/World/Rig").GetPrim()
        self.base     = UsdGeom.Xform.Define(self.stage, "/World/Base").GetPrim()

    def _inherits(self):
        return sorted(str(path) for path in self.usd_prim.get_inherits(self.body))

    def test_add_inherits_accepts_a_prim(self):
        self.usd_prim.add_inherits(self.body, self.rig)
        self.assertEqual(self._inherits(), ["/World/Rig"])

    def test_add_inherits_accepts_a_path_string(self):
        self.usd_prim.add_inherits(self.body, "/World/Rig")
        self.assertEqual(self._inherits(), ["/World/Rig"])

    def test_add_inherits_mixed_list_then_replace(self):
        self.usd_prim.add_inherits(self.body, [self.rig, "/World/Base"])
        self.assertEqual(self._inherits(), ["/World/Base", "/World/Rig"])
        self.usd_prim.set_inherits(self.body, "/World/Base")
        self.assertEqual(self._inherits(), ["/World/Base"])

    def test_clear_inherits(self):
        self.usd_prim.add_inherits(self.body, self.rig)
        self.usd_prim.clear_inherits(self.body)
        self.assertEqual(self._inherits(), [])


if __name__ == "__main__":
    unittest.main()
