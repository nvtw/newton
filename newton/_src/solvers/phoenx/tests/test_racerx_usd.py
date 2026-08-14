# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused USD compatibility tests used by the PhoenX RacerX example."""

from __future__ import annotations

import math
import subprocess
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.examples import example_racerx_usd as racerx
from newton._src.solvers.phoenx.examples import racerx_track

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestRacerXUsd(unittest.TestCase):
    """Verify physics patterns required by the RacerX USD stage."""

    def test_example_supports_direct_file_launch(self) -> None:
        """Allow launching the example by its Python file path."""
        result = subprocess.run(
            [sys.executable, racerx.__file__, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("ImportError", result.stderr)

    def test_vehicle_parts_resolve_across_racerx_models(self) -> None:
        """Resolve A-series, B-series, and C-series vehicles from semantic USD names."""
        for model, chassis_name in (("A1", "Chassis_01"), ("B1", "Chassis_01"), ("C1", "Body_01")):
            with self.subTest(model=model):
                wheel_names = (
                    f"SM_RCCar_{model}_WheelFrontRight_01",
                    f"SM_RCCar_{model}_WheelFrontLeft_01",
                    f"SM_RCCar_{model}_WheelRearRight_01",
                    f"SM_RCCar_{model}_WheelRearLeft_01",
                )
                wheel_paths = tuple(f"/World/RC_Car/{name}" for name in wheel_names)
                chassis_path = f"/World/RC_Car/SM_RCCar_{model}_{chassis_name}"
                servo_path = f"/World/RC_Car/SM_RCCar_{model}_SteeringServo_01"
                wheel_joint_paths = tuple(
                    f"/World/Joints/{corner}/Hinge_Wheel_WheelLinkage" for corner in ("FR", "FL", "RR", "RL")
                )
                steering_path = "/World/Joints/Steering/Steering_Link_Drive"
                shape_paths = tuple(f"{path}/Collision" for path in wheel_paths)
                builder = SimpleNamespace(
                    joint_parent=[4, 4, 4, 4, 5],
                    joint_child=[0, 1, 2, 3, 4],
                    shape_body=[0, 1, 2, 3],
                    shape_type=[newton.GeoType.CONVEX_MESH] * 4,
                    body_mass=[0.05, 0.05, 0.05, 0.05, 3.2, 0.05],
                )
                result = {
                    "path_joint_map": {
                        **dict(zip(wheel_joint_paths, range(4), strict=True)),
                        steering_path: 4,
                    },
                    "path_body_map": {
                        **dict(zip(wheel_paths, range(4), strict=True)),
                        chassis_path: 4,
                        servo_path: 5,
                    },
                    "path_shape_map": dict(zip(shape_paths, range(4), strict=True)),
                }

                parts = racerx._resolve_vehicle_parts(builder, result)

                self.assertEqual(parts.wheel_joints, (0, 1, 2, 3))
                self.assertEqual(parts.wheel_shapes, (0, 1, 2, 3))
                self.assertEqual(parts.wheel_shape_paths, shape_paths)
                self.assertEqual(parts.steering_joint, 4)
                self.assertEqual(parts.chassis_body, 4)
                self.assertEqual(parts.chassis_body_path, chassis_path)
                expected_variant = {"A1": "a3", "B1": "b3", "C1": "c3"}[model]
                self.assertEqual(parts.variant, expected_variant)

    def test_track_follows_closed_uniform_spline(self) -> None:
        """Build a clear, technical circuit with uniformly spaced barriers."""
        half_width = racerx.TRACK_HALF_WIDTH
        spacing = racerx.TRACK_BARRIER_SPACING
        layout = racerx_track.build_track_layout(spacing=spacing, half_width=half_width)
        road_count = len(layout.centerline)

        self.assertGreater(layout.length, 100.0)
        self.assertLess(layout.length, 125.0)
        self.assertEqual(layout.road_poses.shape, (road_count, 7))
        self.assertEqual(layout.barrier_poses.shape, (2 * road_count, 7))
        self.assertEqual(layout.barrier_colors.shape, (2 * road_count, 3))
        self.assertGreater(float(layout.tangents[0, 0]), 0.99)
        self.assertLess(abs(float(layout.tangents[0, 1])), 0.05)

        closed_segments = np.roll(layout.centerline, -1, axis=0) - layout.centerline
        segment_lengths = np.linalg.norm(closed_segments, axis=1)
        self.assertLess(float(segment_lengths.max()), 1.1 * spacing)
        self.assertGreater(float(segment_lengths.min()), 0.79 * spacing)

        tangent_angles = np.arctan2(layout.tangents[:, 1], layout.tangents[:, 0])
        angle_deltas = np.abs(np.angle(np.exp(1j * (np.roll(tangent_angles, -1) - tangent_angles))))
        turn_radii = segment_lengths / np.maximum(angle_deltas, 1.0e-8)
        self.assertLess(float(np.percentile(turn_radii, 10.0)), 1.7)

        indices = np.arange(road_count)
        index_separation = np.abs(indices[:, None] - indices[None, :])
        index_separation = np.minimum(index_separation, road_count - index_separation)
        centerline_distances = np.linalg.norm(layout.centerline[:, None, :] - layout.centerline[None, :, :], axis=2)
        centerline_distances[index_separation <= 10] = np.inf
        barrier_half_width = racerx.TRACK_BARRIER_HALF_EXTENTS[1]
        required_clearance = 2.0 * (half_width + barrier_half_width)
        self.assertGreater(float(centerline_distances.min()), required_clearance)

        left_offsets = layout.barrier_poses[:road_count, :2] - layout.centerline
        right_offsets = layout.barrier_poses[road_count:, :2] - layout.centerline
        np.testing.assert_allclose(np.linalg.norm(left_offsets, axis=1), half_width, atol=1.0e-5)
        np.testing.assert_allclose(np.linalg.norm(right_offsets, axis=1), half_width, atol=1.0e-5)
        np.testing.assert_allclose(left_offsets, -right_offsets, atol=1.0e-5)
        np.testing.assert_allclose(np.linalg.norm(layout.tangents, axis=1), 1.0, atol=1.0e-5)

    def test_track_adds_independent_dynamic_barriers(self) -> None:
        """Add each rainbow barrier box as an independent dynamic body."""
        layout = racerx_track.build_track_layout(
            control_points=np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))),
            spacing=4.0,
        )
        builder = newton.ModelBuilder()

        body_indices = racerx_track.add_track_barriers(
            builder,
            layout,
            half_extents=racerx.TRACK_BARRIER_HALF_EXTENTS,
            density=racerx.TRACK_BARRIER_DENSITY,
        )

        self.assertEqual(len(body_indices), len(layout.barrier_poses))
        self.assertEqual(builder.body_count, len(layout.barrier_poses))
        self.assertEqual(builder.shape_count, len(layout.barrier_poses))
        self.assertTrue(all(flag == int(newton.BodyFlags.DYNAMIC) for flag in builder.body_flags))
        self.assertEqual(racerx.DRIVE_SPEED, 140.0)
        self.assertEqual((racerx.STEERING_TRAVEL, racerx.STEERING_LIMIT), (0.00125, 0.0015))
        self.assertEqual(racerx.STEERING_RATE, 0.015)
        self.assertEqual((racerx.SIM_SUBSTEPS, racerx.SOLVER_ITERATIONS), (4, 6))

    def test_wheel_mesh_colliders_become_symmetric_cylinders(self) -> None:
        """Replace faceted wheel hulls with equally configured smooth cylinders."""
        vertices = np.asarray(
            [
                (-2.0, -0.5, -2.0),
                (-2.0, -0.5, 2.0),
                (-2.0, 0.5, -2.0),
                (-2.0, 0.5, 2.0),
                (2.0, -0.5, -2.0),
                (2.0, -0.5, 2.0),
                (2.0, 0.5, -2.0),
                (2.0, 0.5, 2.0),
            ],
            dtype=np.float32,
        )
        builder = SimpleNamespace(
            shape_type=[newton.GeoType.CONVEX_MESH] * 4,
            shape_source=[SimpleNamespace(vertices=vertices) for _ in range(4)],
            shape_scale=[wp.vec3(0.01, 0.01, 0.01) for _ in range(4)],
            shape_transform=[wp.transform_identity() for _ in range(4)],
            shape_material_mu=[float(index) for index in range(4)],
        )
        wheel_shape_paths = tuple(f"/World/Car/Wheel{index}" for index in range(4))
        parts = racerx._VehicleParts(
            wheel_joints=(0, 1, 2, 3),
            wheel_shapes=(0, 1, 2, 3),
            wheel_shape_paths=wheel_shape_paths,
            steering_joint=4,
            chassis_body=4,
            chassis_body_path="/World/Car/Body",
        )

        racerx._replace_wheel_mesh_colliders(builder, parts)

        self.assertEqual(builder.shape_type, [newton.GeoType.CYLINDER] * 4)
        self.assertEqual(builder.shape_source, [None] * 4)
        np.testing.assert_allclose(
            np.asarray(builder.shape_scale),
            np.tile((0.02, 0.005, 0.0), (4, 1)),
            rtol=0.0,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(builder.shape_material_mu, racerx.WHEEL_FRICTION)
        builder.shape_type = [newton.GeoType.CONVEX_MESH] * 4
        builder.shape_source = [SimpleNamespace(vertices=vertices) for _ in range(4)]
        builder.shape_scale = [wp.vec3(0.01, 0.01, 0.01) for _ in range(4)]
        builder.shape_transform = [wp.transform_identity() for _ in range(4)]
        builder.shape_material_mu = [racerx.WHEEL_FRICTION] * 4
        c3_parts = racerx._VehicleParts(
            wheel_joints=(0, 1, 2, 3),
            wheel_shapes=(0, 1, 2, 3),
            wheel_shape_paths=wheel_shape_paths,
            steering_joint=4,
            chassis_body=4,
            chassis_body_path="/World/Car/Body",
            variant="c3",
        )

        racerx._replace_wheel_mesh_colliders(builder, c3_parts)

        np.testing.assert_allclose(builder.shape_material_mu, racerx.C3_WHEEL_FRICTION)

    def test_looped_vehicle_keeps_friction_only_on_wheels(self) -> None:
        """Keep chassis contacts slippery without removing wheel friction."""
        builder = SimpleNamespace(shape_count=6, shape_material_mu=[0.4, 1.2, 0.6, 1.2, 0.8, 1.2])
        parts = racerx._VehicleParts(
            wheel_joints=(0, 1, 2, 3),
            wheel_shapes=(1, 3, 5, 0),
            wheel_shape_paths=("FR", "FL", "RR", "RL"),
            steering_joint=4,
            chassis_body=4,
            chassis_body_path="/World/RC_Car_B/Body",
            variant="b3",
        )

        racerx._configure_vehicle_ground_friction(builder, parts)

        np.testing.assert_allclose(builder.shape_material_mu, (0.4, 1.2, 0.0, 1.2, 0.0, 1.2))

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
        path_joint_map = {
            f"/World/Joints/{corner}/Slider_Suspension": 4 + index
            for index, corner in enumerate(("FR", "FL", "RR", "RL"))
        }
        parts = racerx._VehicleParts(
            wheel_joints=(0, 1, 2, 3),
            wheel_shapes=(0, 1, 2, 3),
            wheel_shape_paths=("FR", "FL", "RR", "RL"),
            steering_joint=8,
            chassis_body=0,
            chassis_body_path="/World/Car/Body",
        )

        wheel_dofs, steering_joint, steering_dof = racerx._configure_vehicle_joints(
            builder, {"path_joint_map": path_joint_map}, parts
        )

        self.assertEqual(wheel_dofs, [0, 1, 2, 3])
        self.assertEqual(racerx.SUSPENSION_STIFFNESS_SCALE, 0.16)
        self.assertEqual(racerx.STEERING_RATE, 0.015)
        self.assertEqual(racerx.STEERING_STIFFNESS, 64000.0)
        self.assertEqual(racerx.STEERING_DAMPING, 240.0)
        self.assertEqual(racerx.STEERING_FORCE_LIMIT, 320.0)
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

    def test_c3_uses_its_own_rear_suspension_tune(self) -> None:
        """Stiffen only C3's rear suspension while preserving damping ratio."""
        dof_count = 10
        builder = SimpleNamespace(
            joint_qd_start=list(range(9)),
            joint_dof_dim=[(1, 0)] * 8 + [(2, 0)],
            joint_target_mode=[newton.JointTargetMode.NONE] * dof_count,
            joint_target_ke=[0.0] * 4 + [10000.0, 10000.0, 100000.0, 100000.0] + [0.0, 200000.0],
            joint_target_kd=[0.0] * 4 + [1000.0] * 4 + [0.0, 0.0],
            joint_effort_limit=[float("inf")] * dof_count,
            joint_limit_lower=[-1.0] * dof_count,
            joint_limit_upper=[1.0] * dof_count,
        )
        path_joint_map = {
            f"/World/Joints/{corner}/Slider_Suspension": 4 + index
            for index, corner in enumerate(("FR", "FL", "RR", "RL"))
        }
        parts = racerx._VehicleParts(
            wheel_joints=(0, 1, 2, 3),
            wheel_shapes=(0, 1, 2, 3),
            wheel_shape_paths=("FR", "FL", "RR", "RL"),
            steering_joint=8,
            chassis_body=0,
            chassis_body_path="/World/Car/Body",
            variant="c3",
        )

        racerx._configure_vehicle_joints(builder, {"path_joint_map": path_joint_map}, parts)

        rear_scale = racerx.SUSPENSION_STIFFNESS_SCALE * racerx.C3_REAR_SUSPENSION_STIFFNESS_MULTIPLIER
        self.assertEqual(racerx.C3_REAR_SUSPENSION_STIFFNESS_MULTIPLIER, 5.0)
        self.assertEqual(racerx.C3_WHEEL_FRICTION, 1.2)
        self.assertEqual(racerx.C3_SIM_SUBSTEPS, 4)
        for dof in (4, 5):
            self.assertAlmostEqual(builder.joint_target_ke[dof], 10000.0 * racerx.SUSPENSION_STIFFNESS_SCALE)
            self.assertAlmostEqual(builder.joint_target_kd[dof], 1000.0 * math.sqrt(racerx.SUSPENSION_STIFFNESS_SCALE))
        for dof in (6, 7):
            self.assertAlmostEqual(builder.joint_target_ke[dof], 100000.0 * rear_scale)
            self.assertAlmostEqual(builder.joint_target_kd[dof], 1000.0 * math.sqrt(rear_scale))

    def test_b3_normalizes_asymmetric_suspension_gains(self) -> None:
        """Normalize B3's anomalous rear-left spring to its other corners."""
        dof_count = 10
        builder = SimpleNamespace(
            joint_qd_start=list(range(9)),
            joint_dof_dim=[(1, 0)] * 8 + [(2, 0)],
            joint_target_mode=[newton.JointTargetMode.NONE] * dof_count,
            joint_target_ke=[0.0] * 4 + [80000.0, 80000.0, 80000.0, 20000.0] + [0.0, 200000.0],
            joint_target_kd=[0.0] * 4 + [600.0, 600.0, 600.0, 200.0] + [0.0, 0.0],
            joint_effort_limit=[float("inf")] * dof_count,
            joint_limit_lower=[-1.0] * dof_count,
            joint_limit_upper=[1.0] * dof_count,
        )
        path_joint_map = {
            f"/World/Joints_B/{corner}/Slider_Suspension": 4 + index
            for index, corner in enumerate(("FR", "FL", "RR", "RL"))
        }
        parts = racerx._VehicleParts(
            wheel_joints=(0, 1, 2, 3),
            wheel_shapes=(0, 1, 2, 3),
            wheel_shape_paths=("FR", "FL", "RR", "RL"),
            steering_joint=8,
            chassis_body=0,
            chassis_body_path="/World/RC_Car_B/Body",
            variant="b3",
        )

        racerx._configure_vehicle_joints(builder, {"path_joint_map": path_joint_map}, parts)

        expected_ke = 80000.0 * racerx.SUSPENSION_STIFFNESS_SCALE
        expected_kd = 600.0 * math.sqrt(racerx.SUSPENSION_STIFFNESS_SCALE)
        for dof in range(4, 8):
            self.assertAlmostEqual(builder.joint_target_ke[dof], expected_ke)
            self.assertAlmostEqual(builder.joint_target_kd[dof], expected_kd)

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
        launch() if graph is None else wp.capture_launch(graph)

        first_steering_command = racerx.STEERING_RATE / 60.0
        self.assertAlmostEqual(float(target_q.numpy()[9]), first_steering_command, places=7)
        center_to_lock_frames = math.ceil(racerx.STEERING_TRAVEL / first_steering_command)
        for _ in range(center_to_lock_frames - 1):
            launch() if graph is None else wp.capture_launch(graph)
        self.assertAlmostEqual(float(target_q.numpy()[9]), racerx.STEERING_TRAVEL, places=5)

        drive_input.assign(np.asarray((1.0, -1.0), dtype=np.float32))
        lock_to_lock_frames = math.ceil(2.0 * racerx.STEERING_TRAVEL / first_steering_command)
        self.assertLessEqual(lock_to_lock_frames, 10)
        for _ in range(lock_to_lock_frames):
            launch() if graph is None else wp.capture_launch(graph)
        self.assertAlmostEqual(float(target_q.numpy()[9]), -racerx.STEERING_TRAVEL, places=5)

        elapsed_frames = center_to_lock_frames + lock_to_lock_frames
        for _ in range(60 - elapsed_frames):
            launch() if graph is None else wp.capture_launch(graph)

        self.assertAlmostEqual(float(wheel_command.numpy()[0]), racerx.DRIVE_SPEED, places=4)
        self.assertAlmostEqual(float(target_q.numpy()[9]), -racerx.STEERING_TRAVEL, places=5)
        np.testing.assert_allclose(target_qd.numpy()[:4], racerx.DRIVE_SPEED, rtol=0.0, atol=1.0e-4)

        drive_input.zero_()
        for _ in range(30):
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

    def test_chase_camera_kernel_runs_inside_cuda_graph(self) -> None:
        """Publish the level chase camera from a reusable CUDA graph."""
        device = wp.get_device()
        body_q = wp.array(
            [wp.transform((1.0, 2.0, 3.0), wp.quat_identity())],
            dtype=wp.transform,
            device=device,
        )
        initialized = wp.zeros(1, dtype=wp.int32, device=device)
        forwards = wp.empty(1, dtype=wp.vec3, device=device)
        positions = wp.empty(1, dtype=wp.vec3, device=device)
        targets = wp.empty(1, dtype=wp.vec3, device=device)

        def launch():
            wp.launch(
                racerx._update_chase_camera_device,
                dim=1,
                inputs=[body_q, 0, 1.0 / 60.0, 0, initialized],
                outputs=[forwards, positions, targets],
                device=device,
            )

        graph = None
        if device.is_cuda:
            with wp.ScopedCapture() as capture:
                launch()
            graph = capture.graph
        launch() if graph is None else wp.capture_launch(graph)

        np.testing.assert_allclose(
            positions.numpy()[0],
            (1.0 - racerx.CHASE_CAMERA_DISTANCE, 2.0, 3.0 + racerx.CHASE_CAMERA_HEIGHT),
        )
        np.testing.assert_allclose(
            targets.numpy()[0],
            (1.0 + racerx.CHASE_CAMERA_LOOK_AHEAD, 2.0, 3.0 + racerx.CHASE_CAMERA_TARGET_HEIGHT),
        )

        body_q.assign(np.asarray(((10.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32))
        launch() if graph is None else wp.capture_launch(graph)
        np.testing.assert_allclose(
            positions.numpy()[0],
            (10.0 - racerx.CHASE_CAMERA_DISTANCE, 2.0, 3.0 + racerx.CHASE_CAMERA_HEIGHT),
        )
        np.testing.assert_allclose(
            targets.numpy()[0],
            (10.0 + racerx.CHASE_CAMERA_LOOK_AHEAD, 2.0, 3.0 + racerx.CHASE_CAMERA_TARGET_HEIGHT),
        )

    def test_generated_camera_discovers_variant_vehicle_root(self) -> None:
        """Frame a RacerX variant whose vehicle root has a model suffix."""
        stage = Usd.Stage.CreateInMemory()
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        car = UsdGeom.Scope.Define(stage, "/World/RC_Car_B")
        cube = UsdGeom.Cube.Define(stage, "/World/RC_Car_B/Body")
        cube.GetSizeAttr().Set(2.0)

        class Scene:
            def __init__(self):
                self.stage = stage
                self.cameras = []

        class Viewer:
            def __init__(self):
                self.usd_scene = Scene()
                self.camera = None

            def set_camera_look_at(self, position, target, **kwargs):
                self.camera = (position, target, kwargs)

        viewer = Viewer()
        camera_path = racerx._select_authored_camera(viewer, None)

        self.assertEqual(camera_path, "<generated RacerX overview>")
        self.assertIsNotNone(viewer.camera)
        self.assertTrue(car.GetPrim().IsValid())

    def test_looped_vehicle_tire_forces_run_inside_cuda_graph(self) -> None:
        """Apply forward traction and oppose lateral slip inside a CUDA graph."""
        device = wp.get_device()
        body_q = wp.array(
            [wp.transform((0.0, 0.0, 0.0), wp.quat_identity())],
            dtype=wp.transform,
            device=device,
        )
        body_qd = wp.array(
            [wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)],
            dtype=wp.spatial_vector,
            device=device,
        )
        body_f = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        body_indices = wp.array([0], dtype=wp.int32, device=device)
        body_masses = wp.array([2.0], dtype=float, device=device)
        wheel_speed = wp.array([racerx.DRIVE_SPEED], dtype=float, device=device)
        steering = wp.array([racerx.STEERING_TRAVEL], dtype=float, device=device)

        def launch():
            wp.launch(
                racerx._apply_looped_vehicle_tire_forces,
                dim=1,
                inputs=[
                    body_indices,
                    body_masses,
                    0,
                    wp.vec3(1.0, 0.0, 0.0),
                    wheel_speed,
                    steering,
                    body_q,
                    body_qd,
                    body_f,
                ],
                device=device,
            )

        graph = None
        if device.is_cuda:
            with wp.ScopedCapture() as capture:
                launch()
            graph = capture.graph
        launch() if graph is None else wp.capture_launch(graph)

        wrench = body_f.numpy()[0]
        self.assertGreater(float(wrench[0]), 0.0)
        self.assertLess(float(wrench[1]), 0.0)
        self.assertAlmostEqual(float(wrench[5]), 0.0, places=6)

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
