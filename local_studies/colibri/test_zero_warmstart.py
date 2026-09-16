# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check zero-impulse warm starts against the existing ordered accumulation."""

import unittest

import numpy as np
import warp as wp

from local_studies.colibri.zero_warmstart import contact_cached_warmstart_lean as optimized
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactColumnContainer, ContactViews
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import contact_cached_warmstart_lean as reference
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.tests.test_rigid_contact_energy import _make_ordinary_contact


def make_kernel(fast):
    warmstart = optimized if fast else reference

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        body_count: wp.int32,
        cc: ContactContainer,
        contacts: ContactViews,
        copies: CopyStateContainer,
    ):
        warmstart(columns, 0, bodies, particles, body_count, 1000.0, cc, contacts, copies, 0)

    return kernel


class TestZeroWarmstart(unittest.TestCase):
    def test_zero_and_nonzero_impulses(self):
        """Preserve velocity, angular velocity and stored impulses for every zero pattern."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        model, state, pipeline, contacts, solver, _ = _make_ordinary_contact(friction=0.3)
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 1.0e-6)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        world = solver.world
        cc = world._contact_container
        initial_v = world.bodies.velocity.numpy()
        initial_w = world.bodies.angular_velocity.numpy()
        original_impulses = cc.impulses.numpy()
        kernels = [make_kernel(fast) for fast in (False, True)]
        for normal in (0.0, 0.5):
            for tangent1 in (0.0, -0.03):
                for tangent2 in (0.0, 0.02):
                    with self.subTest(normal=normal, tangent1=tangent1, tangent2=tangent2):
                        impulses = original_impulses.copy()
                        impulses[:, 0] = [normal, tangent1, tangent2]
                        results = []
                        for kernel in kernels:
                            world.bodies.velocity.assign(initial_v)
                            world.bodies.angular_velocity.assign(initial_w)
                            cc.impulses.assign(impulses)
                            wp.launch(
                                kernel,
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
                                    world.bodies.velocity.numpy(),
                                    world.bodies.angular_velocity.numpy(),
                                    cc.impulses.numpy(),
                                )
                            )
                        for expected, actual in zip(*results, strict=True):
                            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
