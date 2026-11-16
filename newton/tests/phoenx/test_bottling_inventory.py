# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check source layout inventory transitions without private plant assets."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.examples.phoenx._bottling_scene import _prepare_case_inventory, load_plant
from newton.tests.unittest_utils import requires


@requires("pxr.Usd")
class TestBottlingInventory(unittest.TestCase):
    def test_extended_cartons_preserve_flaps_and_supported_stock(self):
        """Extend finite stock without changing mass, flap ownership or source layers."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, "Z")
        for name in ("C0", "E00"):
            path = f"/Plant/Sim/Product/Cases/PU1_{name}"
            case = UsdGeom.Xform.Define(stage, path)
            case.AddTranslateOp().Set(Gf.Vec3d(0 if name == "C0" else 1, 0, 0.6))
            UsdPhysics.RigidBodyAPI.Apply(case.GetPrim())
            UsdPhysics.MassAPI.Apply(case.GetPrim()).CreateMassAttr(0.42)
            for body in (path, *(f"{path}/Flaps/F{i}" for i in range(4))):
                if body != path:
                    flap = UsdGeom.Xform.Define(stage, body)
                    UsdPhysics.RigidBodyAPI.Apply(flap.GetPrim())
                    UsdPhysics.MassAPI.Apply(flap.GetPrim()).CreateMassAttr(0.02)
                    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/FlapJoints/{body.rsplit('/', 1)[-1]}")
                    joint.CreateBody0Rel().SetTargets([path])
                    joint.CreateBody1Rel().SetTargets([body])
                shape = UsdGeom.Cube.Define(stage, body + "/Collision")
                shape.CreateSizeAttr(0.3)
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        original = stage.GetRootLayer().ExportToString()
        stage.SetEditTarget(stage.GetSessionLayer())
        unchanged, staging = _prepare_case_inventory(stage, None)
        self.assertEqual(unchanged, {"PU1": ["C0", "E00"]})
        self.assertEqual(staging, [])
        inventory, staging = _prepare_case_inventory(stage, 50)
        self.assertEqual(inventory["PU1"], ["C0", *(f"E{i:02}" for i in range(49))])
        self.assertEqual(len(staging), 1)
        self.assertEqual(len(staging[0]["cartons"]), 48)
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        platform = stage.GetPrimAtPath(staging[0]["platform"])
        self.assertTrue(platform.HasAPI(UsdPhysics.CollisionAPI))
        support = bounds.ComputeWorldBound(platform).ComputeAlignedRange()
        ranges = []
        for name in inventory["PU1"]:
            path = f"/Plant/Sim/Product/Cases/PU1_{name}"
            bodies = [p for p in Usd.PrimRange(stage.GetPrimAtPath(path)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            self.assertEqual(len(bodies), 5)
            self.assertAlmostEqual(sum(UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in bodies), 0.5, places=6)
            for i in range(4):
                joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"{path}/FlapJoints/F{i}"))
                self.assertEqual(str(joint.GetBody0Rel().GetTargets()[0]), path)
                self.assertEqual(str(joint.GetBody1Rel().GetTargets()[0]), f"{path}/Flaps/F{i}")
            if name in staging[0]["cartons"]:
                extent = bounds.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
                self.assertAlmostEqual(extent.GetMin()[2] - support.GetMax()[2], 0.0005, places=6)
                self.assertTrue(np.all(np.asarray(extent.GetMin())[:2] > np.asarray(support.GetMin())[:2]))
                self.assertTrue(np.all(np.asarray(extent.GetMax())[:2] < np.asarray(support.GetMax())[:2]))
                for other in ranges:
                    overlap = np.minimum(extent.GetMax(), other.GetMax()) - np.maximum(extent.GetMin(), other.GetMin())
                    self.assertTrue(np.any(overlap[:2] < 0))
                ranges.append(extent)
        self.assertEqual(stage.GetRootLayer().ExportToString(), original)

    def test_wrapper_layout_does_not_block_runtime_inventory(self):
        """Retire layout-only loads and their proxies, preserving real products."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

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
