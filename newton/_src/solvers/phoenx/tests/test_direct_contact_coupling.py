# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for coupled PhoenX direct joints and contact inequalities."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations import direct_contact_gs
from newton._src.solvers.phoenx.examples.example_rigid_cloth_rigid_drop import Example
from newton.viewer import ViewerNull


def _build_multi_world_fixed_bars() -> newton.Model:
    """Build two ill-conditioned rigid bars that land on separate planes."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    template.add_shape_plane(body=-1, width=20.0, length=20.0)
    bodies: list[int] = []
    link_count = 12
    spacing = 0.08
    for index in range(link_count):
        body = template.add_body(
            xform=wp.transform(
                wp.vec3((index - 0.5 * (link_count - 1)) * spacing, 0.0, 0.8),
                wp.quat_identity(),
            )
        )
        template.add_shape_box(
            body,
            hx=0.04,
            hy=0.04,
            hz=0.04,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0 if index % 2 == 0 else 0.1, mu=0.6),
        )
        bodies.append(body)
        if index > 0:
            template.add_joint_fixed(
                parent=bodies[index - 1],
                child=body,
                parent_xform=wp.transform(wp.vec3(spacing, 0.0, 0.0), wp.quat_identity()),
            )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    builder.replicate(template, 2)
    return builder.finalize()


def _build_floating_tree_on_plane() -> newton.Model:
    """Build a synthetic free-root revolute tree above a plane."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.7)
    root = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()))
    child = builder.add_link(xform=wp.transform(wp.vec3(0.4, 0.0, 0.3), wp.quat_identity()))
    builder.add_shape_box(root, hx=0.2, hy=0.1, hz=0.1, cfg=shape_cfg)
    builder.add_shape_box(child, hx=0.2, hy=0.1, hz=0.1, cfg=shape_cfg)
    free_joint = builder.add_joint_free(parent=-1, child=root)
    hinge_joint = builder.add_joint_revolute(
        parent=root,
        child=child,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([free_joint, hinge_joint])
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    builder.color()
    return builder.finalize(device=wp.get_preferred_device())


def _build_closed_chain_on_plane(links: int = 40, plates: int = 1) -> newton.Model:
    """Build a closed loop of ``links`` lying on a plane: one large exact mechanism.

    Each link has ``plates`` side-by-side boxes, so each link touches the plane
    through that many contact manifolds.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.7)
    previous, joints = -1, []
    for i in range(links):
        body = builder.add_link(xform=wp.transform((0.1 * i, 0.0, 0.03), wp.quat_identity()))
        for plate in range(plates):
            offset = 0.06 * (plate - 0.5 * (plates - 1))
            builder.add_shape_box(
                body, xform=wp.transform((0.0, offset, 0.0), wp.quat_identity()), hx=0.05, hy=0.03, hz=0.02, cfg=cfg
            )
        if previous < 0:
            joints.append(builder.add_joint_free(parent=-1, child=body))
        else:
            joints.append(
                builder.add_joint_revolute(
                    previous,
                    body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform((0.05, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform((-0.05, 0.0, 0.0), wp.quat_identity()),
                )
            )
        previous = body
    builder.add_articulation(joints)
    builder.add_joint_revolute(
        previous,
        0,
        axis=newton.Axis.Y,
        parent_xform=wp.transform((0.05, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.1 * links - 0.05, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_ground_plane(cfg=cfg)
    return builder.finalize()


def _build_parallel_steering_on_plane() -> tuple[newton.Model, tuple[int, int], tuple[int, int]]:
    """Build a driven two-wheel steering linkage without external assets."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    identity = wp.quat_identity()
    body_kwargs = {
        "mass": 1.0,
        "inertia": ((0.02, 0.0, 0.0), (0.0, 0.02, 0.0), (0.0, 0.0, 0.02)),
    }
    chassis = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.18), identity), **body_kwargs)
    left_knuckle = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.35, 0.18), identity), **body_kwargs)
    right_knuckle = builder.add_link(xform=wp.transform(wp.vec3(0.0, -0.35, 0.18), identity), **body_kwargs)
    left_wheel = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.35, 0.18), identity), **body_kwargs)
    right_wheel = builder.add_link(xform=wp.transform(wp.vec3(0.0, -0.35, 0.18), identity), **body_kwargs)
    tie_rod = builder.add_link(xform=wp.transform(wp.vec3(-0.12, 0.0, 0.18), identity), **body_kwargs)

    root = builder.add_joint_fixed(
        parent=-1,
        child=chassis,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.18), identity),
        child_xform=wp.transform_identity(),
    )
    steering_joints = []
    wheel_joints = []
    for lateral, knuckle, wheel in (
        (0.35, left_knuckle, left_wheel),
        (-0.35, right_knuckle, right_wheel),
    ):
        steering_joints.append(
            builder.add_joint_revolute(
                parent=chassis,
                child=knuckle,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.0, lateral, 0.0), identity),
                child_xform=wp.transform_identity(),
                target_ke=4000.0,
                target_kd=100.0,
                actuator_mode=newton.JointTargetMode.POSITION,
            )
        )
        wheel_joints.append(
            builder.add_joint_revolute(
                parent=knuckle,
                child=wheel,
                axis=newton.Axis.Y,
                limit_lower=-np.inf,
                limit_upper=np.inf,
                target_ke=40.0,
                target_kd=4.0,
                actuator_mode=newton.JointTargetMode.VELOCITY,
            )
        )

    left_tie = builder.add_joint_ball(
        parent=left_knuckle,
        child=tie_rod,
        parent_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), identity),
        child_xform=wp.transform(wp.vec3(0.0, 0.35, 0.0), identity),
    )
    right_tie = builder.add_joint_ball(
        parent=right_knuckle,
        child=tie_rod,
        parent_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), identity),
        child_xform=wp.transform(wp.vec3(0.0, -0.35, 0.0), identity),
    )
    builder.add_articulation([root, steering_joints[0], wheel_joints[0], steering_joints[1], wheel_joints[1], left_tie])
    builder.joint_articulation[right_tie] = -1

    wheel_cfg = newton.ModelBuilder.ShapeConfig(density=700.0, mu=0.8)
    wheel_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi)
    for wheel in (left_wheel, right_wheel):
        builder.add_shape_cylinder(
            wheel,
            xform=wp.transform(wp.vec3(), wheel_rotation),
            radius=0.18,
            half_height=0.07,
            cfg=wheel_cfg,
        )
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))
    builder.color()
    model = builder.finalize(device=wp.get_preferred_device())
    return model, tuple(steering_joints), tuple(wheel_joints)


class TestDirectContactCoupling(unittest.TestCase):
    def test_rigid_cloth_catches_cube_with_two_inequality_iterations(self):
        """Keep contact effective against a directly constrained mechanism."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        example = Example(ViewerNull(), width=6, height=6)
        direct = example.solver._direct_equality_system
        self.assertEqual(direct.topology.dimensions, (513,))
        self.assertTrue(example.solver.world._skip_all_joint_pgs())

        for _ in range(120):
            example.step()

        example.test_final()
        body_q = example.state.body_q.numpy()
        self.assertTrue(np.isfinite(body_q).all())
        self.assertGreater(float(body_q[example.cube_body, 2]), 0.8)

    def test_mass_splitting_variants_preserve_direct_contact_coupling(self):
        """Couple split contact slots to direct joints in both dispatchers."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        for unrolled in (False, True):
            with self.subTest(unrolled=unrolled):
                example = Example(
                    ViewerNull(),
                    width=4,
                    height=4,
                    mass_splitting=True,
                    mass_splitting_unrolled=unrolled,
                )
                self.assertEqual(example.solver._direct_equality_system.topology.dimensions, (225,))
                self.assertTrue(example.solver.world._skip_all_joint_pgs())

                for _ in range(120):
                    example.step()

                example.test_final()
                cube_z = float(example.state.body_q.numpy()[example.cube_body, 2])
                self.assertGreater(cube_z, 0.8)

    def test_mass_splitting_variants_solve_tree_owned_contacts(self):
        """Solve exact tree contacts in both mass-splitting dispatchers."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        for unrolled in (False, True):
            with self.subTest(unrolled=unrolled):
                model = _build_floating_tree_on_plane()
                pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
                contacts = pipeline.contacts()
                solver = newton.solvers.SolverPhoenX(
                    model,
                    collision_pipeline=pipeline,
                    substeps=5,
                    solver_iterations=2,
                    velocity_iterations=1,
                    joint_mode="maximal_direct",
                    mass_splitting=True,
                    mass_splitting_unrolled=unrolled,
                    step_layout="single_world",
                )
                self.assertTrue(solver._direct_tree_contacts)
                self.assertIsNotNone(solver._maximal_contact_response)
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                control = model.control()
                with wp.ScopedCapture(model.device) as capture:
                    state.clear_forces()
                    pipeline.collide(state, contacts)
                    solver.step(state, state, control, contacts, 1.0 / 60.0)
                for _ in range(120):
                    wp.capture_launch(capture.graph)

                body_q = state.body_q.numpy()
                self.assertTrue(np.isfinite(body_q).all())
                self.assertTrue(np.isfinite(state.body_qd.numpy()).all())
                self.assertGreater(float(np.min(body_q[:, 2])), 0.08)

    def test_mass_splitting_keeps_driven_steering_parallel(self):
        """Keep a closed steering linkage aligned under driven wheel contacts."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        for unrolled in (False, True):
            with self.subTest(unrolled=unrolled):
                model, steering_joints, wheel_joints = _build_parallel_steering_on_plane()
                pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
                contacts = pipeline.contacts()
                solver = newton.solvers.SolverPhoenX(
                    model,
                    collision_pipeline=pipeline,
                    substeps=5,
                    solver_iterations=2,
                    velocity_iterations=1,
                    joint_mode="maximal_direct",
                    mass_splitting=True,
                    mass_splitting_unrolled=unrolled,
                    step_layout="single_world",
                )
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                control = model.control()
                target_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
                qd_start = model.joint_qd_start.numpy()
                for joint in wheel_joints:
                    target_qd[int(qd_start[joint])] = 25.0
                control.joint_target_qd.assign(target_qd)

                with wp.ScopedCapture(model.device) as capture:
                    state.clear_forces()
                    pipeline.collide(state, contacts)
                    solver.step(state, state, control, contacts, 1.0 / 60.0)
                for _ in range(180):
                    wp.capture_launch(capture.graph)

                joint_q = wp.zeros_like(model.joint_q)
                joint_qd = wp.zeros_like(model.joint_qd)
                newton.eval_ik(model, state, joint_q, joint_qd)
                q = joint_q.numpy()
                q_start = model.joint_q_start.numpy()
                steering_angles = np.asarray([q[int(q_start[joint])] for joint in steering_joints])
                steering_angles = (steering_angles + np.pi) % (2.0 * np.pi) - np.pi
                self.assertTrue(np.isfinite(state.body_q.numpy()).all())
                self.assertLess(float(abs(steering_angles[0] - steering_angles[1])), 0.1)

    def test_multi_world_schedulers_couple_direct_bars_to_contacts(self):
        """Keep replicated direct mechanisms above their per-world planes."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        for scheduler in ("fast_tail", "block_world"):
            with self.subTest(scheduler=scheduler):
                model = _build_multi_world_fixed_bars()
                pipeline = newton.CollisionPipeline(model, contact_matching="sticky", broad_phase="nxn")
                contacts = pipeline.contacts()
                solver = newton.solvers.SolverPhoenX(
                    model,
                    collision_pipeline=pipeline,
                    substeps=5,
                    solver_iterations=2,
                    velocity_iterations=1,
                    step_layout="multi_world",
                    multi_world_scheduler=scheduler,
                    joint_mode="maximal_direct",
                )
                self.assertEqual(solver._direct_equality_system.topology.dimensions, (66, 66))
                self.assertTrue(solver.world._skip_all_joint_pgs())
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                control = model.control()
                with wp.ScopedCapture(model.device) as capture:
                    state.clear_forces()
                    pipeline.collide(state, contacts)
                    solver.step(state, state, control, contacts, 1.0 / 60.0)
                for _ in range(120):
                    wp.capture_launch(capture.graph)

                body_q = state.body_q.numpy()
                self.assertTrue(np.isfinite(body_q).all())
                self.assertGreater(float(np.min(body_q[:, 2])), 0.03)

    def test_large_mechanism_sweep_matches_one_warp(self):
        """Sweep a mechanism of more than 32 bodies with several warps and the one-warp result."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        def run(lanes):
            with patch.object(direct_contact_gs, "_COLUMN_LANES", lanes):
                model = _build_closed_chain_on_plane()
                pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
                solver = newton.solvers.SolverPhoenX(
                    model, collision_pipeline=pipeline, substeps=1, solver_iterations=8, joint_mode="maximal_direct"
                )
            self.assertEqual(solver._direct_contact_schedule._lanes, min(lanes, 64))
            state, control, contacts = model.state(), model.control(), pipeline.contacts()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            with wp.ScopedCapture(model.device) as capture:
                pipeline.collide(state, contacts)
                solver.step(state, state, control, contacts, 1.0 / 240.0)
            for _ in range(60):
                wp.capture_launch(capture.graph)
            return state.body_q.numpy()

        several = run(128)
        # Lanes past a manifold's points add exact zeros: the sums keep their order.
        np.testing.assert_array_equal(several, run(32))
        self.assertGreater(float(several[:, 2].min()), 0.015)

    def test_body_pair_manifolds_relax_as_one_run(self):
        """Relax all manifolds of a body pair in one run and keep the chain at rest on the plane."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        model = _build_closed_chain_on_plane(links=40, plates=2)
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model, collision_pipeline=pipeline, substeps=1, solver_iterations=8, joint_mode="maximal_direct"
        )
        state, control, contacts = model.state(), model.control(), pipeline.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        with wp.ScopedCapture(model.device) as capture:
            pipeline.collide(state, contacts)
            solver.step(state, state, control, contacts, 1.0 / 240.0)
        for _ in range(120):
            wp.capture_launch(capture.graph)

        body_q = state.body_q.numpy()
        body_qd = state.body_qd.numpy()
        self.assertTrue(np.isfinite(body_q).all())
        self.assertGreater(float(body_q[:, 2].min()), 0.0195)
        self.assertLess(float(np.abs(body_qd).max()), 0.005)

        schedule = solver._direct_contact_schedule
        total = int(schedule.section_end.numpy()[-1])
        records = schedule.records.numpy()[:total]
        pairs = [tuple(record[:2]) for record in records]
        begin, runs = 0, 0
        while begin < total:
            end = int(records[begin][direct_contact_gs._RECORD_RUN_END])
            self.assertGreater(end, begin)
            self.assertEqual(set(pairs[begin:end]), {pairs[begin]})
            begin, runs = end, runs + 1
        # Both plates of a link are adjacent: one run per link and plane.
        self.assertEqual(runs, len(set(pairs)))
        self.assertLess(runs, total)


if __name__ == "__main__":
    unittest.main()
