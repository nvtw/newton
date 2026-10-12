# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exact mechanisms inside block-PGS joint sweeps (``direct_joint_mask``)."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.block_joint_system import BlockJointSystem, MixedJointSystem
from newton._src.solvers.phoenx.tests.test_contact_coupling import _total_momentum
from newton._src.solvers.phoenx.tests.test_direct_drive import _cuda_with_graph_capture

_LINK_MASSES = (10.0, 0.05, 10.0)
_TARGET_ANGLE = 0.4


def _make_model(count: int = 4) -> newton.Model:
    """Horizontal heavy-light-heavy arms held by stiff position drives, plus free boxes."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for arm in range(count):
        parent = -1
        joints = []
        for link, mass in enumerate(_LINK_MASSES):
            body = builder.add_link(
                xform=wp.transform(wp.vec3(0.5 * link + 0.25, 1.0 * arm, 2.0), wp.quat_identity()),
                mass=mass,
                inertia=wp.mat33(np.eye(3) * 0.01 * mass),
            )
            builder.add_shape_box(body, hx=0.2, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
            joints.append(
                builder.add_joint_revolute(
                    parent,
                    body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(
                        wp.vec3(
                            0.5 * link if parent < 0 else 0.25,
                            1.0 * arm if parent < 0 else 0.0,
                            2.0 if parent < 0 else 0.0,
                        ),
                        wp.quat_identity(),
                    ),
                    child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
                    target_ke=2.0e4,
                    target_kd=50.0,
                    target_pos=0.0,
                    actuator_mode=newton.JointTargetMode.POSITION,
                )
            )
            parent = body
        builder.add_articulation(joints)
    for box in range(6):
        body = builder.add_body(xform=wp.transform(wp.vec3(0.3 * box, -2.0, 0.1 + 0.25 * box), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.color()
    return builder.finalize(device="cuda:0")


def _make_solver(model, **options):
    return newton.solvers.SolverPhoenX(
        model,
        substeps=4,
        solver_iterations=6,
        velocity_iterations=1,
        step_layout="single_world",
        joint_mode="maximal_pgs",
        mass_splitting=True,
        mass_splitting_overflow_only=True,
        mass_splitting_batch_size=1,
        mass_splitting_unrolled=True,
        max_colored_partitions=4,
        partitioner_algorithm="endpoint_owner",
        **options,
    )


def _max_tip_deviation(model, solver, frames=60):
    state = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
    contacts = pipeline.contacts()

    def simulate():
        pipeline.collide(state, contacts)
        solver.step(state, state, control, contacts, 1.0 / 60.0)

    simulate()
    with wp.ScopedCapture() as capture:
        simulate()
    for _ in range(frames):
        wp.capture_launch(capture.graph)
    poses = state.body_q.numpy()
    assert np.isfinite(poses).all()
    tips = poses[2 : 3 * 4 : 3, 2]  # outer heavy link of every arm
    return float(np.max(np.abs(2.0 - tips)))


class TestMixedProjectionOwnership(unittest.TestCase):
    def test_unbounded_exact_mechanisms_require_global_projection(self):
        """Contact dispatch must retain exact equality solves without drives."""
        system = MixedJointSystem.__new__(MixedJointSystem)
        system.exact = SimpleNamespace(enabled=True, has_bounded_drives=False)
        system.blocks = SimpleNamespace(enabled=False, has_bounded_drives=False)
        system._parts = (system.exact,)
        self.assertTrue(system.requires_global_projection)

        system.exact.enabled = False
        system.blocks.enabled = True
        system._parts = (system.blocks,)
        self.assertFalse(system.requires_global_projection)
        system.blocks.has_bounded_drives = True
        self.assertTrue(system.requires_global_projection)


@unittest.skipUnless(_cuda_with_graph_capture(), "PhoenX block PGS requires CUDA graph capture")
class TestMixedJointSystem(unittest.TestCase):
    def test_exact_only_overflow_contacts_conserve_momentum(self):
        """An exact mechanism receives the mean of two unequal contact copies."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        masses = (1.0, 2.0, 3.0, 1.0)
        positions = ((0.0, 0.0, 0.0), (0.999, 0.0, 0.0), (-0.999, 0.0, 0.0), (0.0, 2.0, 0.0))
        bodies = []
        for mass, position in zip(masses, positions, strict=True):
            body = builder.add_body(
                mass=mass,
                inertia=wp.mat33(np.eye(3) * 0.2 * mass),
                xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
            )
            builder.add_shape_sphere(
                body,
                radius=0.5,
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, gap=0.001),
            )
            bodies.append(body)
        builder.add_joint_fixed(
            bodies[0],
            bodies[3],
            parent_xform=wp.transform(wp.vec3(0.0, 1.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, -1.0, 0.0), wp.quat_identity()),
        )
        model = builder.finalize(device="cuda:0")
        state = model.state()
        velocity = state.body_qd.numpy()
        velocity[:, :3] = ((0.0, 0.2, 0.0), (-0.5, 0.4, 0.0), (0.7, -0.2, 0.0), (0.0, 0.2, 0.0))
        state.body_qd.assign(velocity)
        initial = _total_momentum(model, state)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            joint_mode="maximal_pgs",
            direct_joint_mask=np.ones(model.joint_count, dtype=bool),
            step_layout="single_world",
            mass_splitting=True,
            mass_splitting_overflow_only=True,
            mass_splitting_batch_size=1,
            mass_splitting_unrolled=True,
            max_colored_partitions=0,
            contact_chunk_size=1,
            substeps=1,
            solver_iterations=2,
            velocity_iterations=1,
            sor_boost=1.0,
        )
        self.assertFalse(solver._direct_equality_system.blocks.enabled)
        self.assertFalse(solver.world.constraints.bilateral.enabled)
        pipeline.collide(state, contacts)
        self.assertGreaterEqual(int(contacts.rigid_contact_count.numpy()[0]), 2)
        solver.step(state, state, model.control(), contacts, 1.0e-4)
        self.assertGreaterEqual(int(solver.world._copy_state.count_per_node.numpy()[bodies[0] + 1]), 2)
        np.testing.assert_allclose(_total_momentum(model, state), initial, atol=3.0e-6, rtol=0)

    def test_exact_mechanisms_hold_high_mass_ratio_arms(self):
        """Exact mechanisms keep stiff drives rigid across a 200:1 link mass ratio; local blocks do not."""
        model = _make_model()
        mask = np.zeros(model.joint_count, dtype=bool)
        mask[:12] = True
        solver = _make_solver(model, direct_joint_mask=mask, direct_joint_projection_passes=2)
        self.assertIsInstance(solver._direct_equality_system, MixedJointSystem)
        exact_deviation = _max_tip_deviation(model, solver)
        self.assertLess(exact_deviation, 0.02)

        block_model = _make_model()
        self.assertGreater(_max_tip_deviation(block_model, _make_solver(block_model)), 5.0 * exact_deviation)

    def test_mask_and_projection_validation(self):
        """Only block PGS accepts a mask; empty masks keep the plain block system."""
        model = _make_model(1)
        empty = _make_solver(model, direct_joint_mask=np.zeros(model.joint_count, dtype=bool))
        self.assertIsInstance(empty._direct_equality_system, BlockJointSystem)
        with self.assertRaisesRegex(ValueError, "requires joint_mode='maximal_pgs'"):
            newton.solvers.SolverPhoenX(
                model, joint_mode="maximal_direct", direct_joint_mask=np.ones(model.joint_count, dtype=bool)
            )
        with self.assertRaisesRegex(ValueError, "one entry per model joint"):
            _make_solver(model, direct_joint_mask=np.ones(model.joint_count + 1, dtype=bool))
        with self.assertRaisesRegex(ValueError, "direct_joint_mask"):
            _make_solver(model, direct_joint_projection_passes=2)


if __name__ == "__main__":
    unittest.main()
