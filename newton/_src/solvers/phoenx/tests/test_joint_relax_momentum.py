# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Post-integration D6 relaxation must conserve momentum at current poses."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np


class TestJointRelaxMomentum(unittest.TestCase):
    def test_colored_d6_relaxation_uses_current_common_point(self):
        """Moving COMs must not turn a paired internal impulse into a couple."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        for x, mass in ((-0.5, 2.0), (0.5, 3.0)):
            builder.add_link(
                xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                mass=mass,
                inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0),
            )
        builder.add_joint_d6(
            0,
            1,
            parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
            angular_axes=[newton.ModelBuilder.JointDofConfig.create_unlimited(axis) for axis in newton.Axis],
        )
        model = builder.finalize(device="cuda:0", skip_validation_joints=True)
        solver = newton.solvers.SolverPhoenX(
            model,
            joint_mode="maximal_pgs",
            step_layout="single_world",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=1,
            mass_splitting=True,
            mass_splitting_overflow_only=True,
            mass_splitting_unrolled=True,
            mass_splitting_batch_size=1,
            max_colored_partitions=2,
            partitioner_algorithm="endpoint_owner",
            sleeping_velocity_threshold=0.0,
        )
        state = model.state()
        solver.step(state, state, model.control(), None, 1.0e-6)
        world = solver.world
        direct = world._direct_equality_system
        idt = wp.float32(1000.0)
        direct.refresh_geometry(idt)
        direct.prepare_and_factor(idt)
        positions = world.bodies.position.numpy()
        positions[1, 1] += 0.025
        positions[2, 1] -= 0.025
        world.bodies.position.assign(positions)
        velocities = np.zeros((world.num_bodies, 3), dtype=np.float32)
        velocities[1] = [1.0, 0.3, 0.0]
        velocities[2] = [-0.5, -0.2, 0.0]
        world.bodies.velocity.assign(velocities)
        world.bodies.angular_velocity.zero_()
        direct.accumulated_impulse.zero_()
        masses = 1.0 / world.bodies.inverse_mass.numpy()[1:]
        inertia = np.linalg.inv(inertia_sym6_unpack_np(world.bodies.inverse_inertia_world.numpy()[1:]))

        def momentum():
            linear = masses[:, None] * world.bodies.velocity.numpy()[1:].astype(np.float64)
            angular = world.bodies.angular_velocity.numpy()[1:].astype(np.float64)
            return linear.sum(axis=0), (
                np.cross(positions[1:], linear) + np.einsum("bij,bj->bi", inertia, angular)
            ).sum(axis=0)

        before = momentum()
        world._refresh_owned_relax_geometry(idt)
        world._dispatcher.relax(idt)
        after = momentum()
        self.assertGreater(float(np.linalg.norm(direct.accumulated_impulse.numpy())), 0.1)
        np.testing.assert_allclose(after[0], before[0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(after[1], before[1], rtol=0.0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
