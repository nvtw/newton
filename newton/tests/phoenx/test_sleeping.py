# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check sleeping through the public PhoenX solver adapter."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverPhoenX


class TestSleepingKinematicContact(unittest.TestCase):
    def test_moving_support_wakes_stack(self):
        """Wake the whole sleeping stack when its kinematic support rises."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            support = builder.add_body(is_kinematic=True)
            builder.add_shape_box(support, hx=1.0, hy=1.0, hz=0.1)
            cargo = []
            for z in (0.3, 0.7):
                body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, z)))
                builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
                cargo.append(body)
            model = builder.finalize()
            solver = SolverPhoenX(
                model,
                step_layout="single_world",
                substeps=4,
                solver_iterations=4,
                contact_hertz=80,
                sleeping_velocity_threshold=0.02,
                sleeping_frames_required=30,
            )
            state, control = model.state(), model.control()
            contacts = model._collision_pipeline.contacts()
            slots = np.asarray(cargo) + 1
            for _ in range(120):
                solver.wake_on_external_input(state)
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
            self.assertTrue(np.all(solver.world.bodies.island_root.numpy()[slots] >= 0))
            start = state.body_q.numpy()[cargo, 2].copy()
            for step in range(1, 61):
                poses = state.body_q.numpy()
                poses[support, 2] = step * 0.002
                state.body_q.assign(poses)
                solver.wake_on_external_input(state)
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
                self.assertTrue(
                    np.all(solver.world.bodies.island_root.numpy()[slots] < 0),
                    msg=f"moving support failed to wake the stack on step {step}",
                )
            np.testing.assert_allclose(state.body_q.numpy()[cargo, 2] - start, 0.12, atol=0.01)
            for _ in range(120):
                solver.wake_on_external_input(state)
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
            self.assertTrue(np.all(solver.world.bodies.island_root.numpy()[slots] >= 0))
            start = state.body_q.numpy()[cargo, 2].copy()
            initial_velocity = state.body_qd.numpy()[cargo, 2].copy()
            poses = state.body_q.numpy()
            poses[support, 3:] *= -1.0
            state.body_q.assign(poses)
            solver.wake_on_external_input(state)
            self.assertTrue(np.all(solver.world.bodies.island_root.numpy()[slots] >= 0))
            poses[support, 2] = -1.0
            state.body_q.assign(poses)
            solver.wake_on_external_input(state)
            self.assertTrue(
                np.all(solver.world.bodies.island_root.numpy()[slots] < 0),
                msg="removing the support must wake its stack before collision",
            )
            for _ in range(24):
                solver.collide(state, contacts, dt=1 / 120)
                solver.step(state, state, control, contacts, 1 / 120)
            duration = 24 / 120
            expected_fall = initial_velocity * duration - 0.5 * 9.81 * duration**2
            np.testing.assert_allclose(state.body_q.numpy()[cargo, 2] - start, expected_fall, atol=0.01)


if __name__ == "__main__":
    unittest.main()
