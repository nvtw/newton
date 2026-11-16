# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check source layout inventory transitions without private plant assets."""

import tempfile
import unittest
from pathlib import Path

from newton.examples.phoenx._bottling_scene import load_plant
from newton.tests.unittest_utils import USD_AVAILABLE

if USD_AVAILABLE:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestBottlingInventory(unittest.TestCase):
    def test_wrapper_layout_does_not_block_runtime_inventory(self):
        """Retire layout-only loads and their proxies, preserving real products."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            loads = [f"/Plant/WrapLine/Lines/{line}/Loads" for line in ("WL_U", "WL_D")]
            for load in loads:
                UsdGeom.Cube.Define(stage, load + "/Drop/Visual")
            live = UsdGeom.Xform.Define(stage, "/Plant/Logistics/Pallets/Live").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(live)
            UsdPhysics.MassAPI.Apply(live).CreateMassAttr(61.0)
            collider = UsdGeom.Cube.Define(stage, str(live.GetPath()) + "/Collider").GetPrim()
            UsdPhysics.CollisionAPI.Apply(collider)
            prefix = "/Plant/Physics/NavProxies/Replaceable"
            for name, owner in (
                ("layout", ";".join(load + "/Drop/Visual" for load in loads)),
                ("mixed", loads[0] + "/Drop/Visual;/Plant/Other/Visual"),
                ("unrelated", "/Plant/Other/Visual"),
                ("missing_owner", ""),
            ):
                proxy = UsdGeom.Cube.Define(stage, f"{prefix}/{name}").GetPrim()
                UsdPhysics.CollisionAPI.Apply(proxy)
                proxy.CreateAttribute("plant:owner", Sdf.ValueTypeNames.String).Set(owner)
            stage.GetRootLayer().Save()
            original = stage.GetRootLayer().ExportToString()

            scene = load_plant(root, visuals=False, motor_off=True)
            labels = scene.builder.shape_label
            self.assertNotIn(prefix + "/layout/collision", labels)
            for name in ("mixed", "unrelated", "missing_owner"):
                self.assertIn(f"{prefix}/{name}/collision", labels)
            self.assertIn(str(live.GetPath()), scene.builder.body_label)
            self.assertIn(str(collider.GetPath()) + "/collision", labels)
            self.assertEqual(stage.GetRootLayer().ExportToString(), original)


if __name__ == "__main__":
    unittest.main()
