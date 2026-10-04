# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exact mechanisms inside block-PGS joint sweeps (``direct_joint_mask``)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.block_joint_system import BlockJointSystem, MixedJointSystem
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


@unittest.skipUnless(_cuda_with_graph_capture(), "PhoenX block PGS requires CUDA graph capture")
class TestMixedJointSystem(unittest.TestCase):
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
