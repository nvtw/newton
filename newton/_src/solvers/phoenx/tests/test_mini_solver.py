# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Correctness checks for the isolated PhoenX mini experiment."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.mini import MiniSolver, MiniSolverConfig


def _make_stacks(world_count: int, bodies_per_world: int = 4):
    template = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for body_index in range(bodies_per_world):
        body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.48 + 0.96 * body_index), wp.quat_identity()))
        template.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, world_count)
    builder.add_ground_plane()
    return builder.finalize(device=wp.get_preferred_device())


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX mini requires CUDA")
class TestMiniSolver(unittest.TestCase):
    def test_packed_contact_stack_remains_finite(self) -> None:
        model = _make_stacks(32)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", rigid_contact_max=32 * 32, deterministic=True)
        contacts = pipeline.contacts()
        solver = MiniSolver(
            model,
            MiniSolverConfig(substeps=1, iterations=8, max_constraints_per_world=64),
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        for _ in range(60):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
            state_0, state_1 = state_1, state_0

        poses = state_0.body_q.numpy()
        velocities = state_0.body_qd.numpy()
        self.assertTrue(np.isfinite(poses).all())
        self.assertTrue(np.isfinite(velocities).all())
        self.assertGreater(float(poses[:, 2].min()), 0.2)
        self.assertEqual(solver.stats().overflow_constraints, 0)

    def test_interleaved_subwarp_is_bitwise_equivalent(self) -> None:
        model = _make_stacks(16)
        pipelines = [
            newton.CollisionPipeline(model, broad_phase="nxn", rigid_contact_max=16 * 32, deterministic=True)
            for _ in range(2)
        ]
        contacts = [pipeline.contacts() for pipeline in pipelines]
        solvers = [
            MiniSolver(
                model,
                MiniSolverConfig(
                    substeps=1,
                    iterations=4,
                    block_dim=block_dim,
                    max_constraints_per_world=64,
                ),
            )
            for block_dim in (32, 16)
        ]
        states = [(model.state(), model.state()) for _ in solvers]
        control = model.control()

        for _ in range(30):
            for index, (pipeline, solver) in enumerate(zip(pipelines, solvers, strict=True)):
                state_0, state_1 = states[index]
                pipeline.collide(state_0, contacts[index])
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts[index], 1.0 / 120.0)
                states[index] = (state_1, state_0)

        np.testing.assert_array_equal(states[0][0].body_q.numpy(), states[1][0].body_q.numpy())
        np.testing.assert_array_equal(states[0][0].body_qd.numpy(), states[1][0].body_qd.numpy())
        self.assertEqual(solvers[1].stats().overflow_constraints, 0)

    def test_colored_buckets_have_no_body_conflicts(self) -> None:
        model = _make_stacks(16)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", rigid_contact_max=16 * 32, deterministic=True)
        contacts = pipeline.contacts()
        solver = MiniSolver(
            model,
            MiniSolverConfig(substeps=1, iterations=2, max_constraints_per_world=64),
        )
        state_0 = model.state()
        state_1 = model.state()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, model.control(), contacts, 1.0 / 120.0)

        color_counts = solver._world_color_count.numpy()
        world_num_colors = solver._world_num_colors.numpy()
        encoded_constraints = solver._color_constraints.numpy()
        contact_shape0 = contacts.rigid_contact_shape0.numpy()
        contact_shape1 = contacts.rigid_contact_shape1.numpy()
        shape_body = model.shape_body.numpy()
        max_colors = solver.config.max_colors
        color_capacity = solver.config.max_constraints_per_color
        for world in range(model.world_count):
            for color in range(int(world_num_colors[world])):
                color_index = world * max_colors + color
                used_bodies: set[int] = set()
                for slot in range(int(color_counts[color_index])):
                    encoded = int(encoded_constraints[color_index * color_capacity + slot])
                    self.assertGreater(encoded, 0)
                    contact = encoded - 1
                    for shape in (int(contact_shape0[contact]), int(contact_shape1[contact])):
                        body = int(shape_body[shape]) if shape >= 0 else -1
                        if body >= 0:
                            self.assertNotIn(body, used_bodies)
                            used_bodies.add(body)
        self.assertEqual(solver.stats().overflow_constraints, 0)

    def test_mixed_schedule_is_deterministic(self) -> None:
        from newton._src.solvers.phoenx.mini.benchmark import _make_robot_model  # noqa: PLC0415

        device = wp.get_preferred_device()
        model = _make_robot_model(8, 4, str(device))
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            rigid_contact_max=8 * 32,
            deterministic=True,
        )
        contacts = pipeline.contacts()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        pipeline.collide(state, contacts)
        solver = MiniSolver(
            model,
            MiniSolverConfig(substeps=1, iterations=2, max_constraints_per_world=64),
        )
        solver._build_schedule(contacts, color=True)
        expected_counts = solver._world_constraint_count.numpy()
        expected_constraints = solver._world_constraints.numpy()

        for _ in range(10):
            solver._build_schedule(contacts, color=True)
            np.testing.assert_array_equal(solver._world_constraint_count.numpy(), expected_counts)
            np.testing.assert_array_equal(solver._world_constraints.numpy(), expected_constraints)
        self.assertEqual(solver.stats().gather_overflow, 0)
        self.assertEqual(solver.stats().color_overflow, 0)

    def test_packed_revolute_keeps_anchor_bounded(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        link = builder.add_link()
        builder.add_shape_box(link, hx=1.0, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0)),
            child_xform=wp.transform(p=wp.vec3(-1.0, 0.0, 0.0)),
        )
        builder.add_articulation([joint])
        model = builder.finalize(device=wp.get_preferred_device())
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
        contacts = pipeline.contacts()
        solver = MiniSolver(model, MiniSolverConfig(substeps=1, iterations=8))
        state_0 = model.state()
        state_1 = model.state()

        for _ in range(120):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

        pose = state_0.body_q.numpy()[0]
        self.assertTrue(np.isfinite(pose).all())
        self.assertLess(abs(float(np.linalg.norm(pose[:3] - np.array([0.0, 0.0, 2.0]))) - 1.0), 0.15)


if __name__ == "__main__":
    unittest.main()
