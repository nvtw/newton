# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: native preparation erases history at99% of the actual static cone. This asserts the observed premature reset, not acceptance of correct sticking."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    _contact_scatter_colored_rows_kernel,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import (
    contact_prepare_for_iteration_lean_no_soft_pd,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    contact_container_copy_current_to_prev,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_ingest import _contact_warmstart_gather_kernel
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_tangent_delta
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer


@wp.kernel(enable_backward=False)
def _prepare(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    body_count: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    copies: CopyStateContainer,
):
    contact_prepare_for_iteration_lean_no_soft_pd(
        columns, 0, bodies, particles, body_count, wp.float32(1000.0), cc, contacts, copies, 0
    )


class TestRigidFrictionAnchors(unittest.TestCase):
    def test_prepare_preserves_interior_sticking_history(self):
        """Reset a broken material reference without changing collision witnesses."""
        biases = []
        for mass in (1.0,):
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.9999), wp.quat_identity()),
                mass=mass,
                inertia=wp.mat33(mass * 0.4, 0.0, 0.0, 0.0, mass * 0.4, 0.0, 0.0, 0.0, mass * 0.4),
            )
            builder.add_shape_sphere(body, radius=1.0, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))
            builder.add_ground_plane()
            model = builder.finalize(device="cpu")
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=8, contact_matching="sticky")
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                step_layout="single_world",
                substeps=1,
                solver_iterations=1,
                velocity_iterations=0,
                sor_boost=1.0,
            )
            state = model.state()
            pipeline.collide(state, contacts)
            solver.step(state, state, model.control(), contacts, 1.0e-6)
            self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
            # Same stored surface drift and proportionally scaled normal load.
            point = contacts.rigid_contact_point0.numpy().copy()
            point[0, 0] += 1.0e-5
            contacts.rigid_contact_point0.assign(point)
            world = solver.world
            anchor_rows = world._contact_container.lambdas.numpy()
            anchor_rows[6:9, 0] = contacts.rigid_contact_point0.numpy()[0]
            anchor_rows[9:12, 0] = contacts.rigid_contact_point1.numpy()[0]
            world._contact_container.lambdas.assign(anchor_rows)
            impulses = world._contact_container.impulses.numpy().copy()
            impulses[:, 0] = [mass * 1.0e-5, mass * 1.0e-5 * 0.99 * world._contact_cols.data.numpy()[3, 0], 0.0]
            world._contact_container.impulses.assign(impulses)
            wp.launch(
                _prepare,
                dim=1,
                inputs=[
                    world._contact_cols,
                    world.bodies,
                    world.particles or ParticleContainer(),
                    world.num_bodies,
                    world._contact_container,
                    world._active_contact_views(),
                    world._copy_state or CopyStateContainer(),
                ],
                device=model.device,
            )
            biases.append(world._contact_container.derived.numpy()[4:6, 0].copy())
        np.testing.assert_allclose(world._contact_container.impulses.numpy()[1, 0], impulses[1, 0], rtol=0, atol=1e-12)
        self.assertGreater(float(np.linalg.norm(biases[0])), 1e-5)
        wp.launch(_trigger_sliding, 1, [world._contact_container], device=model.device)
        self.assertEqual(float(world._contact_container.lambdas.numpy()[12, 0]), 1.0)
        wp.launch(
            _prepare,
            1,
            [
                world._contact_cols,
                world.bodies,
                world.particles or ParticleContainer(),
                world.num_bodies,
                world._contact_container,
                world._active_contact_views(),
                world._copy_state or CopyStateContainer(),
            ],
            device=model.device,
        )
        self.assertGreater(float(abs(world._contact_container.impulses.numpy()[1, 0])), 0.0)
        self.assertGreater(float(np.linalg.norm(world._contact_container.derived.numpy()[4:6, 0])), 1e-5)
        self.assertEqual(float(world._contact_container.lambdas.numpy()[12, 0]), 1.0)


@wp.kernel
def _tangent_cases(cc: ContactContainer, results: wp.array(dtype=wp.vec2f)):
    i = wp.tid()
    old = wp.float32(0.0)
    rhs = wp.float32(-0.495)
    load = wp.float32(1.0)
    if i == 1:
        rhs = wp.float32(-0.6)
    elif i == 2:
        old = wp.float32(0.49)
        rhs = wp.float32(0.0)
        load = wp.float32(0.8)
    elif i == 3:
        rhs = wp.float32(0.0)
    elif i == 4:
        old = wp.float32(0.5)
        rhs = wp.float32(0.0)
    cc.impulses[1, i] = old
    results[i] = contact_project_tangent_delta(
        cc, i, 0.0, load, rhs, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.4, 1.0, 0.0, 0.0, 0.0
    )


class TestActualProjectionBreak(unittest.TestCase):
    def test_stick_slide_boundary_and_changed_normal_load(self):
        cc = contact_container_zeros(5, device="cpu")
        results = wp.zeros(5, dtype=wp.vec2f, device="cpu")
        wp.launch(_tangent_cases, dim=5, inputs=[cc, results], device="cpu")
        np.testing.assert_allclose(cc.impulses.numpy()[1], [0.495, 0.4, 0.32, 0, 0.5], atol=2e-7, rtol=0)
        self.assertGreaterEqual(cc.lambdas.shape[0], 13, "Explicit actual-projection break history is missing")
        np.testing.assert_array_equal(cc.lambdas.numpy()[12], [0, 1, 1, 0, 0])


@wp.kernel
def _trigger_sliding(cc: ContactContainer):
    cc.impulses[1, 0] = wp.float32(0.0)
    contact_project_tangent_delta(cc, 0, 0.0, 1.0e-5, -1.0e-5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.4, 1.0, 0.0, 0.0, 0.0)


class TestBreakHistoryMatching(unittest.TestCase):
    def test_material_history_follows_matching_and_colored_scatter(self):
        """Preserve reset references through solve order and matched contact IDs."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        for x in (0.0, 4.0):
            body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.9999), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=1.0)
        builder.add_ground_plane()
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=8, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            step_layout="single_world",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
            sor_boost=1.0,
        )
        state = model.state()
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0e-6)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 2)
        world = solver.world
        canonical = world._contact_container
        ordered = contact_container_zeros(8, device=model.device)
        anchor_values = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]], dtype=np.float32
        )
        rows = ordered.lambdas.numpy()
        rows[6:12, :2] = anchor_values
        rows[12, :2] = [1.0, 0.0]
        reference_values = np.arange(28, dtype=np.float32).reshape(14, 2) * 0.001 + 0.123
        self.assertEqual(rows.shape[0], 27)
        rows[13:27, :2] = reference_values
        ordered.lambdas.assign(rows)
        perm = wp.array([1, 0], dtype=wp.int32, device=model.device)
        slots = wp.array([0, 1], dtype=wp.int32, device=model.device)
        count = wp.array([2], dtype=wp.int32, device=model.device)
        matches = wp.array([1, 0, -1, -1, -1, -1, -1, -1], dtype=wp.int32, device=model.device)
        valid_ids = wp.zeros(8, dtype=wp.int32, device=model.device)
        no_reuse = wp.zeros(1, dtype=wp.int32, device=model.device)
        with wp.ScopedCapture(device=model.device) as capture:
            wp.launch(
                _contact_scatter_colored_rows_kernel,
                dim=2,
                inputs=[ordered, canonical, world._contact_cols, slots, perm, slots, count],
                device=model.device,
            )
            contact_container_copy_current_to_prev(canonical, count, device=model.device)
            wp.launch(
                _contact_warmstart_gather_kernel,
                dim=2,
                inputs=[
                    valid_ids,
                    valid_ids,
                    valid_ids,
                    8,
                    matches,
                    valid_ids,
                    no_reuse,
                    0,
                    world.bodies,
                    world._active_contact_views(),
                    0,
                    1,
                    valid_ids,
                    canonical,
                ],
                device=model.device,
            )
        wp.capture_launch(capture.graph)
        # Solve-order scatter reverses the rows; matching reverses them again.
        # Impulse warm-start is disabled, so reference continuity cannot rely on it.
        np.testing.assert_array_equal(canonical.lambdas.numpy()[13:27, 1], reference_values[:, 1])
        self.assertFalse(np.array_equal(canonical.lambdas.numpy()[13:27, 0], reference_values[:, 0]))
        np.testing.assert_array_equal(canonical.lambdas.numpy()[6:12, 1], anchor_values[:, 1])
        self.assertFalse(np.array_equal(canonical.lambdas.numpy()[6:12, 0], anchor_values[:, 0]))
        np.testing.assert_array_equal(canonical.impulses.numpy()[:, :2], 0.0)

        np.testing.assert_array_equal(canonical.lambdas.numpy()[12, :2], [0.0, 0.0])
        matches.assign(np.array([-1, 0, -1, -1, -1, -1, -1, -1], dtype=np.int32))
        wp.capture_launch(capture.graph)
        self.assertEqual(float(canonical.lambdas.numpy()[12, 0]), 0.0)
        self.assertFalse(np.array_equal(canonical.lambdas.numpy()[13:27, 0], reference_values[:, 0]))


@wp.kernel
def _restick_or_unload(cc: ContactContainer, unload: wp.int32):
    i = wp.tid()
    load = wp.float32(1.0)
    if unload != 0:
        load = wp.float32(0.0)
    contact_project_tangent_delta(cc, i, 0.0, load, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.4, 1.0, 0.0, 0.0, 0.0)


class TestResticking(unittest.TestCase):
    def test_latest_solve_can_restick_then_unload(self):
        cc = contact_container_zeros(5, device="cpu")
        results = wp.zeros(5, dtype=wp.vec2f, device="cpu")
        wp.launch(_tangent_cases, 5, [cc, results], device="cpu")
        wp.launch(_restick_or_unload, 5, [cc, 0], device="cpu")
        np.testing.assert_array_equal(cc.lambdas.numpy()[12], [0, 0, 0, 0, 0])
        wp.launch(_restick_or_unload, 5, [cc, 1], device="cpu")
        np.testing.assert_array_equal(cc.lambdas.numpy()[12], [1, 1, 1, 0, 1])
        np.testing.assert_array_equal(cc.impulses.numpy()[1:3], 0.0)
