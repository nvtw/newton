# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conveyor surfaces carry articulations and free bodies in every PhoenX contact path."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton


def _build_belt_scene():
    """Return a model with a free box (articulation 0) and a two-link hinged robot (articulation 1)."""
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.8)
    builder = newton.ModelBuilder()
    box = builder.add_body(xform=wp.transform((0.0, -1.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    root = builder.add_link(xform=wp.transform((0.0, 1.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    foot = builder.add_link(xform=wp.transform((0.3, 1.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(foot, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    free = builder.add_joint_free(root)
    hinge = builder.add_joint_revolute(
        root,
        foot,
        parent_xform=wp.transform((0.15, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform((-0.15, 0.0, 0.0), wp.quat_identity()),
        axis=(0.0, 0.0, 1.0),
    )
    builder.add_articulation([free, hinge])
    vertices = np.array([[-4.0, -3.0, 0.0], [4.0, -3.0, 0.0], [4.0, 3.0, 0.0], [-4.0, 3.0, 0.0]], dtype=np.float32)
    belt = newton.Mesh(
        vertices, np.array([0, 1, 2, 0, 2, 3], dtype=np.int32), compute_inertia=False, enable_surface_velocity=True
    )
    builder.add_shape_mesh(-1, mesh=belt, cfg=cfg)
    return builder.finalize(), belt, [box, root, foot]


@unittest.skipUnless(wp.is_cuda_available(), "conveyor tests use CUDA graph capture")
class TestConveyorArticulations(unittest.TestCase):
    def _assert_bodies_ride_belt(self, **solver_kwargs):
        model, belt, bodies = _build_belt_scene()
        solver = newton.solvers.SolverPhoenX(model, substeps=4, **solver_kwargs)
        state, next_state = model.state(), model.state()
        contacts = model._collision_pipeline.contacts()
        with wp.ScopedCapture() as capture:
            solver.collide(state, contacts)
            solver.step(state, next_state, None, contacts, 1.0 / 60.0)
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                wp.copy(getattr(state, name), getattr(next_state, name))

        for surface_velocity in ((1.0, 0.0, 0.0), (0.0, -0.5, 0.0), (0.0, 0.0, 0.0)):
            belt.mesh.velocities.fill_(wp.vec3(*surface_velocity))
            for _ in range(60):
                wp.capture_launch(capture.graph)
            velocity = state.body_qd.numpy()[bodies]
            np.testing.assert_allclose(velocity[:, :3], np.tile(surface_velocity, (3, 1)), atol=2.0e-2)
            np.testing.assert_allclose(velocity[:, 3:], 0.0, atol=5.0e-2)
            np.testing.assert_allclose(state.body_q.numpy()[bodies, 2], 0.1, atol=3.0e-3)

    def test_reduced_dense_fused_substep(self):
        """The warehouse configuration: reduced robots in the fused dense substep."""
        self._assert_bodies_ride_belt(
            joint_mode="reduced",
            reduced_articulations=[1],
            reduced_backend="dense",
            sleeping_velocity_threshold=0.05,
            solver_iterations=4,
            velocity_iterations=1,
        )

    def test_reduced_articulated_body_backend(self):
        self._assert_bodies_ride_belt(
            joint_mode="reduced", reduced_articulations=[1], reduced_backend="aba", solver_iterations=4
        )

    def test_reduced_single_world(self):
        self._assert_bodies_ride_belt(joint_mode="reduced", reduced_articulations=[1], step_layout="single_world")

    def test_maximal_direct_tree_contacts(self):
        """Mechanism contacts in the direct tree solve, free-body contacts in both layouts."""
        for step_layout in ("multi_world", "single_world"):
            with self.subTest(step_layout=step_layout):
                self._assert_bodies_ride_belt(step_layout=step_layout)

    def test_projected_and_hybrid_trees(self):
        for joint_mode in ("maximal_projected", "hybrid"):
            with self.subTest(joint_mode=joint_mode):
                self._assert_bodies_ride_belt(joint_mode=joint_mode)

    def test_unsupported_paths_raise(self):
        """Paths that would drop the surface velocity refuse the scene."""
        model, _belt, _bodies = _build_belt_scene()
        for kwargs in (
            {"joint_mode": "reduced", "contact_friction_model": "patch"},
            {"step_layout": "single_world", "mass_splitting": True, "colored_contact_headers": True},
        ):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, "Surface-velocity contacts"):
                newton.solvers.SolverPhoenX(model, **kwargs)


if __name__ == "__main__":
    unittest.main()
