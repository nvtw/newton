# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check dispenser stock adaptation without private plant assets."""

import unittest

import numpy as np

import newton
from newton.examples.phoenx._bottling_dispenser import prepare_dispensers
from newton.tests.unittest_utils import requires


@requires("pxr.Usd")
class TestBottlingDispenser(unittest.TestCase):
    def test_restore_existing_stock_and_support(self):
        """Restore source physics without changing stock poses or duplicating visuals."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.CreateInMemory()
        root = "/Plant/WrapLine/Dispensers/EP_PZ1"
        positions = [(0.0, 0.0, 0.0), (0.0, 0.15, 0.2625)]
        for i, position in enumerate(positions):
            pallet = UsdGeom.Xform.Define(stage, f"{root}/Magazine/P{i:02}")
            pallet.AddTranslateOp().Set(position)
            body_path = str(pallet.GetPath()) + "/Geometry/sm_pallet_block_a08_obj_00"
            UsdGeom.Xform.Define(stage, body_path)
            UsdGeom.Cube.Define(stage, body_path + "/Visual")
            collision = pallet.GetPrim().GetVariantSets().AddVariantSet("collision")
            collision.AddVariant("boxes")
            collision.SetVariantSelection("boxes")
            physics = pallet.GetPrim().GetVariantSets().AddVariantSet("physics")
            for name in ("none", "dynamic"):
                physics.AddVariant(name)
            physics.SetVariantSelection("dynamic")
            with physics.GetVariantEditContext():
                body = stage.GetPrimAtPath(body_path)
                UsdPhysics.RigidBodyAPI.Apply(body)
                UsdPhysics.MassAPI.Apply(body).CreateMassAttr(61.0)
                for j in range(2):
                    box = UsdGeom.Cube.Define(stage, body_path + f"/CollisionBoxes/P{j:02}")
                    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            physics.SetVariantSelection("none")

        vertices, indices = [], []
        for center in ((-0.5, -0.3, 0.3375), (-0.5, 0.3, 0.3375), (0.5, -0.3, 0.3375), (0.5, 0.3, 0.3375)):
            mesh = newton.Mesh.create_box(0.125, 0.06, 0.025, compute_inertia=False)
            indices.extend((mesh.indices + len(vertices)).tolist())
            vertices.extend((mesh.vertices + center).tolist())
        steel = UsdGeom.Mesh.Define(stage, root + "/Frame/steel")
        steel.CreatePointsAttr(vertices)
        steel.CreateFaceVertexIndicesAttr(indices)
        steel.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
        physics = UsdGeom.Xform.Define(stage, root + "/Physics")
        physics.AddTranslateOp().Set((0.0, 0.0, 0.1))
        material = UsdShade.Material.Define(stage, "/Materials/Static")
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateDynamicFrictionAttr(0.45)
        tower = stage.DefinePrim(root + "/Physics/Static/tower_l")
        UsdShade.MaterialBindingAPI.Apply(tower).Bind(material, materialPurpose="physics")
        owners = {
            "covered": root + "/Magazine/P00/Geometry/sm_pallet_block_a08_obj_00/Visual",
            "unrelated": "/Plant/WrapLine/Lines/WL_U/Loads/Pick/Geometry",
            "mixed": root + "/Magazine/P01/Geometry/sm_pallet_block_a08_obj_00/Visual; /Plant/Uncovered/Geometry",
            "missing": "",
        }
        for name, owner in owners.items():
            proxy = stage.DefinePrim("/Plant/Physics/NavProxies/Replaceable/" + name)
            proxy.CreateAttribute("plant:owner", Sdf.ValueTypeNames.String).Set(owner)
        source = stage.GetRootLayer().ExportToString()
        paths = prepare_dispensers(stage)
        self.assertEqual(len(paths), 2)
        self.assertEqual(stage.GetRootLayer().ExportToString(), source)
        for name in owners:
            proxy = stage.GetPrimAtPath("/Plant/Physics/NavProxies/Replaceable/" + name)
            self.assertEqual(proxy.IsActive(), name != "covered")
        for i, body_path in enumerate(paths):
            body = stage.GetPrimAtPath(body_path)
            self.assertTrue(body.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertEqual(UsdPhysics.MassAPI(body).GetMassAttr().Get(), 61.0)
            self.assertEqual(len(body.GetChild("CollisionBoxes").GetChildren()), 2)
            np.testing.assert_allclose(
                UsdGeom.XformCache().GetLocalToWorldTransform(body).ExtractTranslation(), positions[i]
            )
        for i in range(4):
            finger = stage.GetPrimAtPath(root + f"/Physics/SeparatorFinger{i}")
            self.assertTrue(finger.HasAPI(UsdPhysics.CollisionAPI))
            self.assertEqual(finger.GetAttribute("visibility").Get(), "invisible")
            points = np.array(finger.GetAttribute("points").Get())
            points[:, 2] += 0.1
            np.testing.assert_allclose(points, np.asarray(vertices[i * 24 : (i + 1) * 24]), atol=1.0e-7)
            self.assertEqual(
                UsdShade.MaterialBindingAPI(finger).ComputeBoundMaterial("physics")[0].GetPath(), material.GetPath()
            )
        prepare_dispensers(stage)
        self.assertEqual(len(stage.GetPrimAtPath(root + "/Magazine").GetChildren()), 2)
        self.assertEqual(len(stage.GetPrimAtPath(root + "/Physics").GetChildren()), 5)


if __name__ == "__main__":
    unittest.main()
