# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check finite-effort drives coupled to hard joint stops."""

import unittest

import numpy as np
import warp as wp

import newton


class TestDriveLimitConsistency(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
