# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check finite-effort drives coupled to hard joint stops."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton


class TestDriveLimitConsistency(unittest.TestCase):
    def test_exact_hinge_with_reserved_joints_retains_limits(self):
        """Reserve runtime bonds without duplicating exact rows or losing hard stops."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=builder.ShapeConfig(density=0))
            hinge = builder.add_joint_revolute(-1, body, axis=newton.Axis.Z, limit_lower=-0.1, limit_upper=0.1)
            builder.add_articulation([hinge])
            builder.reserve_dynamic_capacity(joints=1)
            model = builder.finalize()
            exact = np.zeros(model.joint_count, dtype=bool)
            exact[hinge] = True
            solver = newton.solvers.SolverPhoenX(
                model,
                joint_mode="maximal_pgs",
                step_layout="single_world",
                direct_joint_mask=exact,
                solver_iterations=4,
                substeps=1,
            )
            cid = int(solver._joint_constraints.joint_idx_to_cid.numpy()[hinge])
            # The unilateral angular stops must remain active in the colored solve.
            self.assertEqual(solver.world._joint_pgs_enabled.numpy()[cid], 1)
            state, control = model.state(), model.control()
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[body, 5] = 4.0
            state.body_f.assign(force)
            for _ in range(120):
                solver.step(state, state, control, None, 1.0 / 240)
            np.testing.assert_allclose(state.joint_q.numpy()[0], 0.1, atol=1e-4)
            np.testing.assert_allclose(state.body_q.numpy()[body, :3], 0, atol=1e-6)
            self.assertTrue(np.isfinite(state.body_qd.numpy()).all())

    def test_force_limited_drives_hold_hard_stops(self):
        """Reach and hold hard stops without recoiling when motor effort saturates."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            for _ in range(2):
                body = builder.add_link(mass=0.35, inertia=wp.mat33(np.eye(3) * 0.001))
                joint = builder.add_joint_prismatic(
                    -1,
                    body,
                    axis=newton.Axis.X,
                    limit_lower=0.0,
                    limit_upper=0.065,
                    target_kd=6666.6665,
                    effort_limit=150.0,
                    actuator_mode=newton.JointTargetMode.VELOCITY,
                )
                builder.add_articulation([joint])
            model = builder.finalize()
            # Both external loads are smaller than the motor's stall effort.
            # One opposes travel; the other increases the stop reaction.
            forces = np.array([[-2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]], dtype=np.float32)
            for direct in (False, True):
                for iterations in (4, 64):
                    with self.subTest(direct=direct, iterations=iterations):
                        state, control = model.state(), model.control()
                        control.joint_target_qd.fill_(0.45)
                        state.body_f.assign(forces)
                        solver = newton.solvers.SolverPhoenX(
                            model,
                            joint_mode="maximal_pgs",
                            step_layout="single_world",
                            solver_iterations=iterations,
                            substeps=1,
                            velocity_iterations=1,
                            direct_joint_mask=np.ones(model.joint_count, dtype=bool) if direct else None,
                        )
                        previous = state.joint_q.numpy().copy()
                        # At 0.45 m/s these 65 mm strokes reach their stops in
                        # roughly 0.15 s; the remaining steps verify holding.
                        for step in range(96):
                            solver.step(state, state, control, None, 1.0 / 240.0)
                            position = state.joint_q.numpy().copy()
                            self.assertTrue(np.all(np.isfinite(position)))
                            self.assertGreaterEqual(
                                float(np.min(position - np.minimum(previous, 0.065))),
                                -1.0e-5,
                                f"A driven slider recoiled at step {step + 1}: {previous} -> {position}",
                            )
                            # Finite sweeps can leave some penetration; recovery
                            # may remove that error but must not retract below
                            # the stop against the continuously extending motor.
                            self.assertLessEqual(float(position.max()), 0.065 + 0.45 / 240.0)
                            if step >= 47:
                                np.testing.assert_allclose(position, 0.065, atol=1.0e-5)
                            previous = position

    @unittest.skipUnless(wp.get_cuda_device_count(), "requires CUDA graph capture")
    def test_loaded_saturated_drive_keeps_its_active_set(self):
        """Hold a stalled motor at its cap without re-solving the same bound every substep."""
        from newton._src.solvers.phoenx.articulations.direct_equality import DirectEqualitySystem  # noqa: PLC0415

        @wp.kernel
        def count_bound_changes(flag: wp.array[wp.int32], changes: wp.array[wp.int32]):
            changes[0] += flag[0]

        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            builder.add_shape_box(-1, xform=wp.transform((0.3, 0, 0), wp.quat_identity()), hx=0.1, hy=0.2, hz=0.2)
            slider = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
            builder.add_shape_box(slider, hx=0.1, hy=0.1, hz=0.1)
            joint = builder.add_joint_prismatic(
                -1,
                slider,
                axis=newton.Axis.X,
                target_kd=1000.0,
                effort_limit=5.0,
                actuator_mode=newton.JointTargetMode.VELOCITY,
            )
            builder.add_articulation([joint])
            model = builder.finalize()
            solver = newton.solvers.SolverPhoenX(
                model,
                joint_mode="maximal_pgs",
                step_layout="single_world",
                substeps=2,
                solver_iterations=4,
                direct_joint_mask=np.ones(model.joint_count, dtype=bool),
            )
            state, control = model.state(), model.control()
            control.joint_target_qd.fill_(0.5)
            contacts = solver._collision_pipeline.contacts()
            changes = wp.zeros(1, dtype=wp.int32)
            find_new_bounds = DirectEqualitySystem._find_new_bounds

            def counted_scan(system, idt, flag):
                find_new_bounds(system, idt, flag)
                wp.launch(count_bound_changes, dim=1, inputs=[system.drive_active_set_flag, changes])

            def step():
                solver.collide(state, contacts, dt=1.0 / 120.0)
                solver.step(state, state, control, contacts, 1.0 / 120.0)

            with patch.object(DirectEqualitySystem, "_find_new_bounds", counted_scan):
                step()
                with wp.ScopedCapture() as capture:
                    step()
            for _ in range(120):
                wp.capture_launch(capture.graph)
            # The slider stalls against the wall, so corrections ran. They must
            # consume their flags: scans rely on finding them cleared.
            self.assertGreater(int(changes.numpy()[0]), 0)
            system = solver.world._direct_equality_system
            system = getattr(system, "exact", system)
            np.testing.assert_array_equal(system._bound_scan_flags.numpy(), 0)
            changes.zero_()
            for _ in range(60):
                wp.capture_launch(capture.graph)
            # The wall carries the full capped motor force once the slider stalls.
            self.assertEqual(int(changes.numpy()[0]), 0)
            np.testing.assert_allclose(state.body_qd.numpy()[slider], 0.0, atol=1.0e-3)
            np.testing.assert_allclose(state.body_q.numpy()[slider, 0], 0.1, atol=2.0e-3)


if __name__ == "__main__":
    unittest.main()
