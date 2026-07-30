# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for coupled PhoenX direct joints and contact inequalities."""

import unittest

import numpy as np
import warp as wp

import newton
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
                    articulation_mode="maximal",
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


if __name__ == "__main__":
    unittest.main()
