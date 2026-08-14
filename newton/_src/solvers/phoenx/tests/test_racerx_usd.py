# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused USD compatibility tests used by the PhoenX RacerX example."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.examples import example_racerx_usd as racerx

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestRacerXUsd(unittest.TestCase):
    """Verify physics patterns required by the RacerX USD stage."""

    def test_coaxial_hard_limit_and_spring_merge_to_one_dof(self) -> None:
        """Merge stacked travel and spring joints into one driven coordinate."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")

        parent = UsdGeom.Cube.Define(stage, "/World/Parent")
        child = UsdGeom.Cube.Define(stage, "/World/Child")
        for body in (parent, child):
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())

        travel = self._add_prismatic(stage, "/World/Travel", parent, child, -0.9, 0.4)
        spring = self._add_prismatic(stage, "/World/Spring", parent, child, 0.0, 0.01)
        spring.GetPrim().CreateAttribute(
            "physxLimit:linear:stiffness",
            Sdf.ValueTypeNames.Float,
        ).Set(80000.0)
        spring.GetPrim().CreateAttribute(
            "physxLimit:linear:damping",
            Sdf.ValueTypeNames.Float,
        ).Set(600.0)

        builder = newton.ModelBuilder()
        result = builder.add_usd(
            stage,
            load_visual_shapes=False,
            schema_resolvers=[newton.usd.SchemaResolverPhysx()],
        )

        self.assertEqual(builder.joint_count, 1)
        self.assertEqual(builder.joint_dof_dim, [(1, 0)])
        self.assertEqual(result["path_joint_map"][str(travel.GetPath())], 0)
        self.assertEqual(result["path_joint_map"][str(spring.GetPath())], 0)
        self.assertAlmostEqual(builder.joint_limit_lower[0], -0.9)
        self.assertAlmostEqual(builder.joint_limit_upper[0], 0.4)
        self.assertAlmostEqual(builder.joint_target_q[0], 0.005)
        self.assertAlmostEqual(builder.joint_target_ke[0], 80000.0)
        self.assertAlmostEqual(builder.joint_target_kd[0], 600.0)
        self.assertEqual(builder.joint_target_mode[0], newton.JointTargetMode.POSITION)

    def test_vehicle_controls_bound_steering_and_soften_suspension(self) -> None:
        """Configure finite steering effort and visible suspension travel."""
        dof_count = 10
        builder = SimpleNamespace(
            joint_qd_start=list(range(9)),
            joint_dof_dim=[(1, 0)] * 8 + [(2, 0)],
            joint_target_mode=[newton.JointTargetMode.NONE] * dof_count,
            joint_target_ke=[0.0] * 4 + [80000.0] * 4 + [0.0, 200000.0],
            joint_target_kd=[0.0] * 4 + [600.0] * 4 + [0.0, 0.0],
            joint_effort_limit=[float("inf")] * dof_count,
            joint_limit_lower=[-1.0] * dof_count,
            joint_limit_upper=[1.0] * dof_count,
        )
        path_joint_map = {path: index for index, path in enumerate(racerx.WHEEL_JOINT_PATHS)}
        path_joint_map.update(
            {
                f"/World/Joints/{corner}/Slider_Suspension": 4 + index
                for index, corner in enumerate(("FR", "FL", "RR", "RL"))
            }
        )
        path_joint_map[racerx.STEERING_JOINT_PATH] = 8

        wheel_dofs, steering_joint, steering_dof = racerx._configure_vehicle_joints(
            builder, {"path_joint_map": path_joint_map}
        )

        self.assertEqual(wheel_dofs, [0, 1, 2, 3])
        self.assertEqual(racerx.SUSPENSION_STIFFNESS_SCALE, 0.16)
        self.assertEqual(racerx.STEERING_RATE, 0.03)
        self.assertEqual(racerx.STEERING_STIFFNESS, 8000.0)
        self.assertEqual(racerx.STEERING_DAMPING, 80.0)
        self.assertEqual(racerx.STEERING_FORCE_LIMIT, 80.0)
        for dof in wheel_dofs:
            self.assertEqual(builder.joint_target_mode[dof], newton.JointTargetMode.VELOCITY)
            self.assertEqual(builder.joint_effort_limit[dof], racerx.DRIVE_TORQUE_LIMIT)
        for dof in range(4, 8):
            self.assertAlmostEqual(
                builder.joint_target_ke[dof],
                80000.0 * racerx.SUSPENSION_STIFFNESS_SCALE,
            )
            self.assertAlmostEqual(
                builder.joint_target_kd[dof],
                600.0 * math.sqrt(racerx.SUSPENSION_STIFFNESS_SCALE),
            )
        self.assertEqual(steering_joint, 8)
        self.assertEqual(steering_dof, 9)
        self.assertEqual(builder.joint_target_ke[steering_dof], racerx.STEERING_STIFFNESS)
        self.assertEqual(builder.joint_target_kd[steering_dof], racerx.STEERING_DAMPING)
        self.assertEqual(builder.joint_effort_limit[steering_dof], racerx.STEERING_FORCE_LIMIT)
        self.assertEqual(builder.joint_limit_lower[steering_dof], -racerx.STEERING_LIMIT)
        self.assertEqual(builder.joint_limit_upper[steering_dof], racerx.STEERING_LIMIT)

    def test_vehicle_control_kernel_ramps_inside_cuda_graph(self) -> None:
        """Ramp motor and steering targets inside a reusable CUDA graph."""
        device = wp.get_device()
        drive_input = wp.array([1.0, 1.0], dtype=float, device=device)
        wheel_dofs = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
        wheel_command = wp.zeros(1, dtype=float, device=device)
        steering_command = wp.zeros(1, dtype=float, device=device)
        target_q = wp.zeros(10, dtype=float, device=device)
        target_qd = wp.zeros(10, dtype=float, device=device)

        def launch():
            wp.launch(
                racerx._update_vehicle_controls,
                dim=1,
                inputs=[
                    drive_input,
                    wheel_dofs,
                    9,
                    1.0 / 60.0,
                    wheel_command,
                    steering_command,
                    target_q,
                    target_qd,
                ],
                device=device,
            )

        graph = None
        if device.is_cuda:
            with wp.ScopedCapture() as capture:
                launch()
            graph = capture.graph
        for _ in range(30):
            launch() if graph is None else wp.capture_launch(graph)

        self.assertAlmostEqual(float(wheel_command.numpy()[0]), racerx.DRIVE_SPEED, places=4)
        self.assertAlmostEqual(float(target_q.numpy()[9]), racerx.STEERING_TRAVEL, places=5)
        np.testing.assert_allclose(target_qd.numpy()[:4], racerx.DRIVE_SPEED, rtol=0.0, atol=1.0e-4)

        drive_input.zero_()
        for _ in range(15):
            launch() if graph is None else wp.capture_launch(graph)
        self.assertAlmostEqual(float(wheel_command.numpy()[0]), 0.0, places=4)

    def test_chase_camera_stays_level_behind_chassis(self) -> None:
        """Place the chase camera behind the chassis without inheriting roll."""
        pose = np.asarray((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)

        eye, target = racerx._chase_camera_targets(pose)

        np.testing.assert_allclose(
            eye,
            (1.0 - racerx.CHASE_CAMERA_DISTANCE, 2.0, 3.0 + racerx.CHASE_CAMERA_HEIGHT),
        )
        np.testing.assert_allclose(
            target,
            (1.0 + racerx.CHASE_CAMERA_LOOK_AHEAD, 2.0, 3.0 + racerx.CHASE_CAMERA_TARGET_HEIGHT),
        )

    def test_self_filtered_collision_group_excludes_member_pair(self) -> None:
        """Translate a self-filtered USD collision group into shape exclusions."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")

        members = []
        for name in ("A", "B"):
            body = UsdGeom.Cube.Define(stage, f"/World/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())
            members.append(body.GetPath())

        group = UsdPhysics.CollisionGroup.Define(stage, "/World/CollisionGroup")
        collection = Usd.CollectionAPI.Apply(group.GetPrim(), "colliders")
        collection.CreateIncludesRel().SetTargets(members)
        group.CreateFilteredGroupsRel().SetTargets([group.GetPath()])

        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)

        self.assertEqual(builder.shape_count, 2)
        self.assertIn((0, 1), builder.shape_collision_filter_pairs)

    @staticmethod
    def _add_prismatic(stage, path, parent, child, lower, upper):
        """Author one X-axis prismatic joint with explicit anchors."""
        joint = UsdPhysics.PrismaticJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([parent.GetPath()])
        joint.CreateBody1Rel().SetTargets([child.GetPath()])
        joint.CreateAxisAttr().Set("X")
        joint.CreateLowerLimitAttr().Set(lower)
        joint.CreateUpperLimitAttr().Set(upper)
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        return joint


if __name__ == "__main__":
    unittest.main()
