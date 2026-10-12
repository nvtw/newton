# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Topology-sized joint tiles retain the generic block solve arithmetic."""

import functools
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import BodyContainer, inertia_sym6_unpack_np
from newton._src.solvers.phoenx.constraints.bilateral_joint import (
    get_bilateral_joint_tile_width,
    get_iterate_bilateral_joint_block,
)
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer


@functools.cache
def _make_sweep(row_capacity):
    iterate = get_iterate_bilateral_joint_block(True, row_capacity=row_capacity)

    @wp.kernel(module="unique", enable_backward=False)
    def run(
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        copies: CopyStateContainer,
        cid: wp.int32,
    ):
        iterate(constraints, cid, bodies, particles, copies, bodies.position.shape[0], -1, True, wp.tid())

    return run


def _make_model(kinds):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for index, kind in enumerate(kinds):
        pair = []
        for x, mass in ((-0.5, 2.0), (0.5, 3.0)):
            pair.append(
                builder.add_link(
                    xform=wp.transform(wp.vec3(x, 2.0 * index, 0.0), wp.quat_identity()),
                    mass=mass,
                    inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0),
                )
            )
        options = {
            "parent_xform": wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            "child_xform": wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
        }
        if kind == "sphere":
            builder.add_joint_d6(
                *pair,
                **options,
                angular_axes=[newton.ModelBuilder.JointDofConfig.create_unlimited(axis) for axis in newton.Axis],
            )
        elif kind == "fixed":
            builder.add_joint_fixed(*pair, **options)
        else:
            builder.add_joint_revolute(
                *pair,
                **options,
                axis=newton.Axis.Z,
                target_ke=40.0 if kind == "drive" else 0.0,
                target_kd=1.0 if kind == "drive" else 0.0,
                target_pos=0.2,
                actuator_mode=newton.JointTargetMode.POSITION,
            )
    return builder.finalize(device="cuda:0", skip_validation_joints=True)


class TestJointRowCapacity(unittest.TestCase):
    def test_topology_capacity_and_generic_parity(self):
        """Exercise spherical, hinge, fixed, and mixed driven joint blocks."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        for kinds, expected in ((["sphere"], 3), (["hinge"], 6), (["fixed"], 6), (["sphere", "drive"], 6)):
            with self.subTest(kinds=kinds):
                model = _make_model(kinds)
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
                solver.step(state, state, model.control(), None, 1e-6)
                world = solver.world
                data = world.constraints.bilateral
                self.assertEqual(int(data.row_capacity), expected)
                self.assertEqual(get_bilateral_joint_tile_width(expected), 4 if expected == 3 else 8)
                expected_counts = [{"sphere": 3, "hinge": 5, "fixed": 6, "drive": 6}[kind] for kind in kinds]
                np.testing.assert_array_equal(data.row_count.numpy(), expected_counts)
                rotations = world.bodies.orientation.numpy()
                for body in range(1, world.num_bodies):
                    rotations[body] = np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, 3)), 0.1 * body))
                world.bodies.orientation.assign(rotations)
                world._refresh_world_inertia()
                direct = world._direct_equality_system
                direct.begin_substep(wp.float32(1000.0))
                direct.prepare_and_factor(wp.float32(1000.0))
                rng = np.random.default_rng(42)
                v = rng.normal(size=(world.num_bodies, 3)).astype(np.float32)
                w = rng.normal(size=(world.num_bodies, 3)).astype(np.float32)
                v[0] = 0
                w[0] = 0
                positions = world.bodies.position.numpy()[1:].astype(np.float64)
                masses = 1.0 / world.bodies.inverse_mass.numpy()[1:]
                inertia = np.linalg.inv(inertia_sym6_unpack_np(world.bodies.inverse_inertia_world.numpy()[1:]))

                def momentum(velocity, omega, masses=masses, positions=positions, inertia=inertia):
                    linear = masses[:, None] * velocity[1:].astype(np.float64)
                    return np.r_[
                        linear.sum(axis=0),
                        (np.cross(positions, linear) + np.einsum("bij,bj->bi", inertia, omega[1:])).sum(axis=0),
                    ]

                reference = None
                for capacity in (6, expected, expected):
                    world.bodies.velocity.assign(v)
                    world.bodies.angular_velocity.assign(w)
                    data.accumulated.zero_()
                    for cid in range(model.joint_count):
                        wp.launch(
                            _make_sweep(capacity),
                            dim=4 if capacity == 3 else 8,
                            block_dim=32,
                            inputs=[
                                world.constraints,
                                world.bodies,
                                world._particles_or_sentinel(),
                                world._copy_state,
                                cid,
                            ],
                            device=model.device,
                        )
                    result = (
                        world.bodies.velocity.numpy(),
                        world.bodies.angular_velocity.numpy(),
                        data.accumulated.numpy(),
                    )
                    # Driven rows can apply large opposing impulses; account
                    # for their FP32 velocity scatter before checking parity.
                    np.testing.assert_allclose(momentum(*result[:2]), momentum(v, w), rtol=0.0, atol=1e-5)
                    if reference is not None:
                        for actual, original in zip(result, reference, strict=True):
                            np.testing.assert_array_equal(actual, original)
                    reference = result


if __name__ == "__main__":
    unittest.main()
