# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check dispenser stock adaptation without private plant assets."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx._bottling_dispenser import (
    dispense_strokes,
    dispenser_unit,
    kinematic_strokes,
    prepare_dispensers,
)
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

        store = "/Plant/WrapLine/Stores/EPS_U"
        for i in range(2):
            stock = stage.DefinePrim(f"{store}/bay_1_front/P{i:02}")
            stock.GetReferences().AddInternalReference(f"{root}/Magazine/P{i:02}")
            stock.GetVariantSet("physics").SetVariantSelection("none")
        for name in ("bay_1_front", "bay_1_back", "guard"):
            proxy = UsdGeom.Cube.Define(stage, store + "/Physics/" + name)
            UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())

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
        plate = UsdGeom.Cube.Define(stage, root + "/Physics/Static/push_out")
        plate.AddTranslateOp().Set((0.0, 0.9, 0.07))
        UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
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
        for i in range(2):
            stock = stage.GetPrimAtPath(f"{store}/bay_1_front/P{i:02}/Geometry/sm_pallet_block_a08_obj_00")
            self.assertTrue(stock.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertEqual(len(stock.GetChild("CollisionBoxes").GetChildren()), 2)
        self.assertFalse(stage.GetPrimAtPath(store + "/Physics/bay_1_front").IsActive())
        self.assertTrue(stage.GetPrimAtPath(store + "/Physics/bay_1_back").IsActive())
        self.assertTrue(stage.GetPrimAtPath(store + "/Physics/guard").IsActive())
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
        # The left and right finger pairs ride two kinematic carriages; the push-out plate a third.
        for i in range(4):
            carriage = stage.GetPrimAtPath(root + f"/Physics/Separator{'LR'[i // 2]}")
            self.assertTrue(UsdPhysics.RigidBodyAPI(carriage).GetKinematicEnabledAttr().Get())
            finger = carriage.GetChild(f"Finger{i % 2}")
            self.assertTrue(finger.HasAPI(UsdPhysics.CollisionAPI))
            self.assertEqual(finger.GetAttribute("visibility").Get(), "invisible")
            points = np.array(finger.GetAttribute("points").Get())
            points[:, 2] += 0.1
            np.testing.assert_allclose(points, np.asarray(vertices[i * 24 : (i + 1) * 24]), atol=1.0e-7)
            self.assertEqual(
                UsdShade.MaterialBindingAPI(finger).ComputeBoundMaterial("physics")[0].GetPath(), material.GetPath()
            )
        pusher = stage.GetPrimAtPath(root + "/Physics/PushOut")
        self.assertTrue(UsdPhysics.RigidBodyAPI(pusher).GetKinematicEnabledAttr().Get())
        self.assertTrue(pusher.GetChild("plate").HasAPI(UsdPhysics.CollisionAPI))
        self.assertFalse(plate.GetPrim().IsActive())
        plate_center = UsdGeom.XformCache().GetLocalToWorldTransform(pusher.GetChild("plate")).ExtractTranslation()
        np.testing.assert_allclose(plate_center, (0.0, 0.9, 0.17), atol=1.0e-7)
        prepare_dispensers(stage)
        self.assertEqual(len(stage.GetPrimAtPath(root + "/Magazine").GetChildren()), 2)
        self.assertEqual(len(stage.GetPrimAtPath(root + "/Physics").GetChildren()), 4)


def _block_pallet(builder, xy, z, cfg):
    """Add a 1.0 x 1.2 x 0.1625 m block pallet whose side pockets are open along x, as the source A08."""
    body = builder.add_body(xform=wp.transform((*xy, z), wp.quat_identity()))
    for height, half in ((0.01, 0.01), (0.1525, 0.01)):  # bottom boards and top deck
        builder.add_shape_box(
            body, xform=wp.transform((0, 0, height), wp.quat_identity()), hx=0.5, hy=0.6, hz=half, cfg=cfg
        )
    for y in (-0.55, 0.0, 0.55):  # block rows between the pockets
        builder.add_shape_box(
            body, xform=wp.transform((0, y, 0.08125), wp.quat_identity()), hx=0.5, hy=0.05, hz=0.06125, cfg=cfg
        )
    return body


class TestDispenserSequence(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
    def test_dispense_presents_the_next_pallet_at_pick(self):
        """Lower, release, re-clamp and lift the column, then push its bottom pallet into the pick opening."""
        # The source EP_PZ1 dispenser, with its pick opening at the origin.
        spec = {
            "ports": {"pick": {"xy": [0.0, 0.0]}, "load": {"xy": [0.0, 0.15]}},
            "magazine": {"pallet_z": [0.0, 0.2625, 0.425], "column_behind_pick_m": 0.15},
            "timers_s": {"dispense_s": 15.0},
        }
        unit = dispenser_unit(spec)
        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder()
            builder.add_ground_plane(cfg=builder.ShapeConfig(mu=0.5))
            wood = builder.ShapeConfig(density=250.0, mu=0.5)
            steel = builder.ShapeConfig(density=0.0, mu=0.3)
            # The separator's finger pairs and the push-out plate, in their source poses.
            carriages = []
            for side in (-1.0, 1.0):
                carriage = builder.add_body(is_kinematic=True)
                for y in (-0.28, 0.28):
                    pose = wp.transform((side * 0.505, 0.15 + y, 0.3375), wp.quat_identity())
                    builder.add_shape_box(carriage, xform=pose, hx=0.125, hy=0.06, hz=0.025, cfg=steel)
                carriages.append(carriage)
            pusher = builder.add_body(is_kinematic=True)
            pose = wp.transform((0.0, 0.905, 0.07), wp.quat_identity())
            builder.add_shape_box(pusher, xform=pose, hx=0.3, hy=0.145, hz=0.07, cfg=steel)
            carriages.append(pusher)
            column = [_block_pallet(builder, (0.0, 0.15), z, wood) for z in (0.2625, 0.425, 0.5875)]
            model = builder.finalize()
            solver = newton.solvers.SolverPhoenX(model, substeps=4, solver_iterations=8, step_layout="single_world")
            state, control = model.state(), model.control()
            contacts = model._collision_pipeline.contacts()
            bodies = wp.array(carriages, dtype=int)
            rest = wp.array(state.body_q.numpy()[carriages], dtype=wp.transform)
            strokes = wp.zeros(3, dtype=wp.vec3)
            dt = 1.0 / 60.0
            with wp.ScopedCapture() as capture:
                wp.launch(kinematic_strokes, dim=3, inputs=[bodies, rest, strokes, state.body_q])
                solver.collide(state, contacts, dt=dt)
                solver.step(state, state, control, contacts, dt)
            # Let the column settle on the fingers, dispense, then let it settle again.
            for _frame in range(60):
                wp.capture_launch(capture.graph)
            hanging = state.body_q.numpy()[column[0], 2]
            for frame in range(60 * 17):
                strokes.assign(np.asarray(dispense_strokes(unit, frame * dt), dtype=np.float32))
                wp.capture_launch(capture.graph)
            poses = state.body_q.numpy()[column]

        # The column's bottom pallet stands seated and square in the pick opening.
        np.testing.assert_allclose(poses[0, :3], (0.0, 0.0, 0.0), atol=0.03)
        tilt = np.degrees(2.0 * np.arcsin(np.linalg.norm(poses[:, 3:5], axis=1)))
        self.assertTrue(np.all(tilt < 2.0))
        # The fingers hold the next pallet at the column's height again, with the rest stacked on it.
        self.assertGreater(hanging, 0.15)
        self.assertAlmostEqual(float(poses[1, 2]), float(hanging), delta=0.01)
        np.testing.assert_allclose(poses[1:, 1], 0.15, atol=0.03)
        self.assertAlmostEqual(float(poses[2, 2] - poses[1, 2]), 0.1625, delta=0.01)


if __name__ == "__main__":
    unittest.main()
