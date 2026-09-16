# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check exact inactive-friction elision against general prepared contact rows."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_get_body1,
    contact_get_body2,
    contact_set_friction,
    contact_set_friction_dynamic,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import _make_contact_iterate_at
from newton._src.solvers.phoenx.constraints.constraint_container import constraint_bodies_make
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.tests.test_rigid_contact_energy import _make_ordinary_contact


@wp.kernel(enable_backward=False)
def _set_material(columns: ContactColumnContainer, mu_s: wp.float32, mu_k: wp.float32):
    contact_set_friction(columns, 0, mu_s)
    contact_set_friction_dynamic(columns, 0, mu_k)


def _make_sweep(*, fast, bias, pd):
    factory = _make_contact_iterate_at
    iterate = factory(
        cloth_support=False,
        has_mass_splitting=False,
        use_bias=bias,
        has_soft_contact_pd=pd,
        frictionless_fast_path=False,
        normal_first=fast,
    )

    @wp.kernel(enable_backward=False)
    def sweep(
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        body_count: wp.int32,
        cc: ContactContainer,
        contacts: ContactViews,
        copies: CopyStateContainer,
    ):
        pair = constraint_bodies_make(contact_get_body1(columns, 0), contact_get_body2(columns, 0))
        iterate(
            columns,
            0,
            0,
            bodies,
            particles,
            body_count,
            pair,
            wp.float32(1000.0),
            cc,
            contacts,
            copies,
            0,
            wp.float32(1.0),
        )

    return sweep


class TestNormalFirstContact(unittest.TestCase):
    def test_normal_first_matches_general_contact_equations(self):
        """Preserve impulses and velocities for PD, speculation and stale friction."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        for friction, dynamic_friction in ((0.0, 0.0), (0.3, 0.3), (0.3, 0.0), (0.0, 0.3)):
            model, state, pipeline, contacts, solver, _ = _make_ordinary_contact(friction=friction)
            pipeline.collide(state, contacts)
            solver.step(state, state, model.control(), contacts, 1.0e-6)
            self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
            world = solver.world
            wp.launch(
                _set_material,
                dim=1,
                inputs=[world._contact_cols, friction, dynamic_friction],
                device=model.device,
            )
            cc = world._contact_container
            initial_v = world.bodies.velocity.numpy()
            initial_w = world.bodies.angular_velocity.numpy()
            initial_impulses = cc.impulses.numpy()
            initial_derived = cc.derived.numpy()
            for bias in (False, True):
                for pd in (False, True):
                    for normal_seed in (0.0, 0.5):
                        for stale_tangent in (False, True):
                            for separation in (-0.0001, 0.001):
                                with self.subTest(
                                    friction=friction,
                                    bias=bias,
                                    pd=pd,
                                    stale_tangent=stale_tangent,
                                    separation=separation,
                                ):
                                    derived = initial_derived.copy()
                                    derived[3, 0] = separation * 1000.0
                                    derived[4:6, 0] = [0.01, -0.02]
                                    if pd:
                                        derived[6:9, 0] = [10.0, 0.2, 0.001]
                                    impulses = initial_impulses.copy()
                                    impulses[:, 0] = [
                                        normal_seed,
                                        0.03 if stale_tangent else 0.0,
                                        -0.02 if stale_tangent else 0.0,
                                    ]
                                    results = []
                                    for fast in (False, True):
                                        world.bodies.velocity.assign(
                                            initial_v if normal_seed else np.zeros_like(initial_v)
                                        )
                                        world.bodies.angular_velocity.assign(
                                            initial_w if normal_seed else np.zeros_like(initial_w)
                                        )
                                        cc.impulses.assign(impulses)
                                        cc.derived.assign(derived)
                                        wp.launch(
                                            _make_sweep(fast=fast, bias=bias, pd=pd),
                                            dim=1,
                                            inputs=[
                                                world._contact_cols,
                                                world.bodies,
                                                world.particles or ParticleContainer(),
                                                world.num_bodies,
                                                cc,
                                                world._active_contact_views(),
                                                world._copy_state or CopyStateContainer(),
                                            ],
                                            device=model.device,
                                        )
                                        results.append(
                                            (
                                                cc.impulses.numpy(),
                                                world.bodies.velocity.numpy(),
                                                world.bodies.angular_velocity.numpy(),
                                            )
                                        )
                                    if any(not np.array_equal(a, b) for a, b in zip(*results, strict=True)):
                                        print(
                                            "CASE",
                                            friction,
                                            dynamic_friction,
                                            bias,
                                            pd,
                                            normal_seed,
                                            stale_tangent,
                                            separation,
                                        )
                                        print("GENERAL", [a[:, :3] for a in results[0]])
                                        print("SPLIT", [a[:, :3] for a in results[1]])
                                    for general, optimized in zip(*results, strict=True):
                                        np.testing.assert_array_equal(optimized, general)
                                    if friction == 0.0 and dynamic_friction == 0.0 and (bias or separation < 0.0):
                                        np.testing.assert_array_equal(results[1][0][1:3, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
