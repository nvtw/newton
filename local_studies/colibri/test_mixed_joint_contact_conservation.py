# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Conserve momentum across real grouped contact solve and relaxation phases."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np
from newton._src.solvers.phoenx.tests.test_direct_drive import _cuda_with_graph_capture


def _physical_totals(world):
    """Sum physical body momentum and energy, independently of copy storage."""
    inverse_mass = world.bodies.inverse_mass.numpy().astype(np.float64)
    active = inverse_mass > 0.0
    mass = 1.0 / inverse_mass[active]
    position = world.bodies.position.numpy()[active].astype(np.float64)
    velocity = world.bodies.velocity.numpy()[active].astype(np.float64)
    spin = world.bodies.angular_velocity.numpy()[active].astype(np.float64)
    inertia = np.linalg.inv(
        inertia_sym6_unpack_np(world.bodies.inverse_inertia_world.numpy()[active]).astype(np.float64)
    )
    linear = mass[:, None] * velocity
    angular = np.cross(position, linear) + np.einsum("bij,bj->bi", inertia, spin)
    energy = 0.5 * (np.sum(linear * velocity) + np.sum(spin * np.einsum("bij,bj->bi", inertia, spin)))
    return np.r_[linear.sum(axis=0), angular.sum(axis=0)], float(energy)


@unittest.skipUnless(_cuda_with_graph_capture(), "Color groups require CUDA")
class TestMixedJointContactConservation(unittest.TestCase):
    def test_mixed_joint_contacts_conserve_through_relaxation(self):
        """Preserve momentum with a revolute block and friction sharing copied bodies."""
        for reverse in (False, True):
            for width in (1, 4):
                with self.subTest(reverse=reverse, width=width):
                    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                    positions = (-1.0000001, 0.0, 1.0000001)
                    if reverse:
                        positions = positions[::-1]
                    for x in positions:
                        mass = 2.0 + x * 0.5
                        body = builder.add_link(
                            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                            mass=mass,
                            inertia=wp.mat33(0.2 * mass, 0.0, 0.0, 0.0, 0.3 * mass, 0.0, 0.0, 0.0, 0.4 * mass),
                        )
                        builder.add_shape_box(
                            body,
                            hx=0.5,
                            hy=0.5,
                            hz=0.5,
                            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, gap=0.001),
                        )
                    direction = positions[1] - positions[0]
                    root = builder.add_joint_free(0)
                    hinge = builder.add_joint_revolute(
                        0,
                        1,
                        axis=newton.Axis.Z,
                        parent_xform=wp.transform(wp.vec3(direction / 2, 0.0, 0.0), wp.quat_identity()),
                        child_xform=wp.transform(wp.vec3(-direction / 2, 0.0, 0.0), wp.quat_identity()),
                    )
                    builder.add_articulation([root, hinge])
                    builder.add_articulation([builder.add_joint_free(2)])
                    model = builder.finalize(device="cuda:0")
                    state = model.state()
                    qd = state.body_qd.numpy()
                    for index, x in enumerate(positions):
                        qd[index, :3] = [-0.5 * x + 0.2, 0.2 * x + 0.3, 0.05]
                        qd[index, 3:] = [0.25, -0.3, 0.4]
                    state.body_qd.assign(qd)
                    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="sticky")
                    contacts = pipeline.contacts()
                    solver = newton.solvers.SolverPhoenX(
                        model,
                        collision_pipeline=pipeline,
                        articulation_mode="maximal",
                        joint_solver="block_pgs",
                        step_layout="single_world",
                        mass_splitting=True,
                        mass_splitting_color_group_size=width,
                        contact_chunk_size=1,
                        parallel_contact_prepare=True,
                        substeps=1,
                        solver_iterations=2,
                        velocity_iterations=1,
                        sor_boost=1.0,
                    )
                    world = solver.world
                    dispatcher_type = type(world._dispatcher)
                    original_solve, original_relax = dispatcher_type.solve, dispatcher_type.relax
                    records = []
                    poses = []
                    activity = []

                    def observed(original, phase, world=world, records=records, poses=poses, activity=activity):
                        def call(dispatcher, idt):
                            before, energy_before = _physical_totals(world)
                            velocity_before = world.bodies.velocity.numpy().copy()
                            original(dispatcher, idt)
                            after, energy_after = _physical_totals(world)
                            activity.append(
                                (phase, float(np.max(np.abs(world.bodies.velocity.numpy() - velocity_before))))
                            )
                            records.append((phase, before, after, energy_before, energy_after))
                            poses.append(
                                (world.bodies.position.numpy().copy(), world.bodies.orientation.numpy().copy())
                            )

                        return call

                    with (
                        patch.object(dispatcher_type, "solve", observed(original_solve, "biased")),
                        patch.object(dispatcher_type, "relax", observed(original_relax, "relax")),
                    ):
                        for _ in range(4):
                            state.clear_forces()
                            pipeline.collide(state, contacts)
                            solver.step(state, state, model.control(), contacts, 0.001)
                    self.assertGreater(float(np.max(np.abs(world.constraints.bilateral.accumulated.numpy()))), 1e-5)
                    self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                    self.assertEqual([r[0] for r in records], ["biased", "relax"] * 4)
                    counts = world._copy_state.count_per_node.numpy()
                    dynamic_counts = counts[world.bodies.inverse_mass.numpy() > 0.0]
                    self.assertGreater(int(dynamic_counts.max()), int(dynamic_counts.min()))
                    self.assertGreaterEqual(int(dynamic_counts.max()), 2)
                    self.assertGreater(float(np.max(np.abs(poses[1][0] - poses[0][0]))), 1e-5)
                    self.assertGreater(float(np.max(np.abs(poses[1][1] - poses[0][1]))), 1e-5)
                    for phase in ("biased", "relax"):
                        self.assertGreater(max(change for name, change in activity if name == phase), 1e-5)
                    for phase, before, after, _, _ in records:
                        np.testing.assert_allclose(after, before, atol=3e-6, rtol=0.0, err_msg=phase)
                    np.testing.assert_allclose(records[-1][2], records[0][1], atol=5e-6, rtol=0.0)
                    self.assertLessEqual(records[-1][4], records[0][3] + 5e-6)


if __name__ == "__main__":
    unittest.main()
