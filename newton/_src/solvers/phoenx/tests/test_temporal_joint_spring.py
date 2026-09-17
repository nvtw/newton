# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check temporal force-spring increments and paired angular momentum."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.bilateral_joint import get_iterate_bilateral_joint_block
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_get_body1,
    constraint_get_body2,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.tests.test_block_joint_policy import make_model, make_solver


@wp.kernel
def reset_hinge(constraints: ConstraintContainer, bodies: BodyContainer):
    a = constraint_get_body1(constraints, 0)
    b = constraint_get_body2(constraints, 0)
    bodies.angular_velocity[a] = wp.vec3f(0.0, 0.0, -0.1)
    bodies.angular_velocity[b] = wp.vec3f(0.0, 0.0, 0.1)


def make_kernel(cooperative, temporal, use_bias=True):
    iterate = get_iterate_bilateral_joint_block(cooperative, temporal_springs=temporal)

    @wp.kernel(module="unique", enable_backward=False)
    def run(
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        copies: CopyStateContainer,
    ):
        lane = wp.tid()
        iterate(constraints, 0, bodies, particles, copies, bodies.position.shape[0], 0, use_bias, lane)

    return run


class TestTemporalJointSpring(unittest.TestCase):
    def test_joint_correction_uses_full_temporal_timestep(self):
        """Apply the source joint coefficient once and preserve paired spatial rows."""
        if not wp.is_cuda_available():
            self.skipTest("Joint fixture requires CUDA")
        with wp.ScopedDevice("cuda:0"):
            model = make_model(0.0)
            solver = make_solver(model, mass_splitting=False)
            state = model.state()
            solver.step(state, state, model.control(), None, 0.01)
            world = solver.world
            direct = world._direct_equality_system
            positions = world.bodies.position.numpy()
            positions[2] = [0.01, 0.02, -0.03]
            world.bodies.position.assign(positions)
            orientations = world.bodies.orientation.numpy()
            orientations[2] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.1))
            world.bodies.orientation.assign(orientations)
            direct.refresh_geometry(100.0)
            original_bias = direct.row_bias.numpy()
            for substeps in (1, 24, 30, 100):
                direct.set_temporal_substeps(substeps)
                direct.refresh_geometry(100.0)
                count = int(direct.row_count.numpy()[0])
                error = direct.row_error.numpy()[0, :count]
                self.assertGreater(float(np.linalg.norm(error)), 0.01)
                coefficient = 0.5 * min(0.9, 2.0 / substeps**0.5)
                np.testing.assert_allclose(
                    direct.row_bias.numpy()[0, :count], error * coefficient * 100.0, rtol=2e-6, atol=1e-7
                )
                j0 = direct.row_wrench0.numpy()[0, :count]
                j1 = direct.row_wrench1.numpy()[0, :count]
                np.testing.assert_allclose(j0[:, :3] + j1[:, :3], 0, atol=2e-7)
                torque = j0[:, 3:] + j1[:, 3:] + np.cross(positions[1], j0[:, :3]) + np.cross(positions[2], j1[:, :3])
                np.testing.assert_allclose(torque, 0, atol=2e-7)
            direct.set_temporal_substeps(None)
            direct.refresh_geometry(100.0)
            np.testing.assert_array_equal(direct.row_bias.numpy(), original_bias)

    def test_relaxation_does_not_integrate_springs(self):
        """Relax structural velocities without applying another spring impulse."""
        if not wp.is_cuda_available():
            self.skipTest("Cooperative solve requires CUDA")
        with wp.ScopedDevice("cuda:0"):
            model = make_model(40.0)
            solver = make_solver(model, mass_splitting=False)
            control = model.control()
            control.joint_target_q.assign([1.0])
            state = model.state()
            solver.step(state, state, control, None, 0.01)
            world = solver.world
            data = world.constraints.bilateral
            row = int(np.flatnonzero(data.row_dynamic.numpy())[0])
            for cooperative in (False, True):
                with self.subTest(cooperative=cooperative):
                    world.bodies.velocity.assign([[0, 0, 0], [0.2, 0, 0], [-0.2, 0, 0]])
                    world.bodies.angular_velocity.assign([[0, 0, 0], [0.1, 0, -0.1], [-0.1, 0, 0.1]])
                    accumulated = np.zeros(data.accumulated.shape, dtype=np.float32)
                    accumulated[row] = 0.2
                    data.accumulated.assign(accumulated)
                    wp.launch(
                        make_kernel(cooperative, True, False),
                        8 if cooperative else 1,
                        [world.constraints, world.bodies, world._particles_or_sentinel(), world._copy_state],
                        block_dim=8 if cooperative else 1,
                        device=world.device,
                    )
                    self.assertEqual(data.accumulated.numpy()[row], accumulated[row])
                    velocity = world.bodies.velocity.numpy()
                    angular = world.bodies.angular_velocity.numpy()
                    np.testing.assert_allclose(velocity[1] - velocity[2], 0, atol=2e-7)
                    np.testing.assert_allclose(angular[1, :2] - angular[2, :2], 0, atol=2e-7)
                    np.testing.assert_allclose(angular[1:, 2], [-0.1, 0.1], atol=2e-7)

    def test_temporal_increment_preserves_both_momenta(self):
        """Match a force-spring step independently of earlier accumulated increments."""
        if not wp.is_cuda_available():
            self.skipTest("Cooperative solve requires CUDA")
        with wp.ScopedDevice("cuda:0"):
            model = make_model(40.0)
            solver = make_solver(model, mass_splitting=False)
            control = model.control()
            control.joint_target_q.assign([1.0])
            state = model.state()
            solver.step(state, state, control, None, 0.01)
            world = solver.world
            data = world.constraints.bilateral
            dynamic = np.flatnonzero(data.row_dynamic.numpy())
            self.assertEqual(len(dynamic), 1)
            row = int(dynamic[0])
            a = float(data.dynamic_mass.numpy()[row])
            reference = float(data.reference.numpy()[row])
            # h * (d + h*k), with h=.01, d=0, k=40.
            self.assertAlmostEqual(a, 0.004, delta=1e-8)
            for cooperative in (False, True):
                for temporal in (False, True):
                    run = make_kernel(cooperative, temporal)
                    for old_impulse in (0.0, 0.2):
                        with self.subTest(cooperative=cooperative, temporal=temporal, old=old_impulse):
                            world.bodies.velocity.zero_()
                            world.bodies.angular_velocity.zero_()
                            wp.launch(reset_hinge, 1, [world.constraints, world.bodies], device=world.device)
                            accumulated = np.zeros(data.accumulated.shape, dtype=np.float32)
                            accumulated[row] = old_impulse
                            data.accumulated.assign(accumulated)
                            before = world.bodies.angular_velocity.numpy()
                            wp.launch(
                                run,
                                8 if cooperative else 1,
                                [world.constraints, world.bodies, world._particles_or_sentinel(), world._copy_state],
                                block_dim=8 if cooperative else 1,
                                device=world.device,
                            )
                            compliance = 0.0 if temporal else old_impulse / a
                            expected_delta = (reference - 0.2 - compliance) / (2.0 / 0.8 + 1.0 / a)
                            actual_delta = float(data.accumulated.numpy()[row]) - old_impulse
                            self.assertAlmostEqual(actual_delta, expected_delta, delta=3e-8)
                            after = world.bodies.angular_velocity.numpy()
                            np.testing.assert_allclose(after.sum(axis=0), before.sum(axis=0), atol=2e-7)
                            np.testing.assert_array_equal(world.bodies.velocity.numpy(), 0)
                            changed = after[:, 2] - before[:, 2]
                            np.testing.assert_allclose(
                                np.sort(changed), np.sort([-expected_delta / 0.8, 0, expected_delta / 0.8]), atol=2e-7
                            )


if __name__ == "__main__":
    unittest.main()
