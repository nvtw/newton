# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check zero-impulse warm starts against the existing ordered accumulation."""

import unittest

import numpy as np
import warp as wp

from local_studies.colibri.zero_column_warmstart import _contact_cached_warmstart_split as optimized
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactColumnContainer, ContactViews
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import _contact_cached_warmstart_split as reference
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer, copy_state_container_zeros
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
        copies = copy_state_container_zeros(2, world.num_bodies, device=model.device)
        rng = np.random.default_rng(1847)
        cv = rng.normal(size=(2, 3)).astype(np.float32)
        cw = rng.normal(size=(2, 3)).astype(np.float32)
        headers = world._contact_cols.data.numpy()
        for slots in ((-1, -1), (0, -1), (-1, 1), (0, 1)):
            for length in (1, 3, 7):
                h = headers.copy()
                hi = h.view(np.int32)
                hi[6, 0] = length
                hi[7, 0], hi[8, 0] = slots
                hi[9, 0] = 3 if slots[0] >= 0 else 1
                hi[10, 0] = 5 if slots[1] >= 0 else 1
                world._contact_cols.data.assign(h)
                for field in (cc.lambdas, cc.derived):
                    values = field.numpy()
                    values[:, :length] = values[:, 0:1]
                    field.assign(values)
                for pattern in range(8):
                    impulses = original_impulses.copy()
                    impulses[:, :length] = 0
                    for k in range(length):
                        if k % 3 != 1:
                            impulses[:, k] = [
                                0.0003 if pattern & 1 else 0,
                                -0.00003 if pattern & 2 else 0,
                                0.00002 if pattern & 4 else 0,
                            ]
                    results = []
                    for kernel in kernels:
                        world.bodies.velocity.assign(initial_v)
                        world.bodies.angular_velocity.assign(initial_w)
                        copies.velocity.assign(cv)
                        copies.angular_velocity.assign(cw)
                        cc.impulses.assign(impulses)
                        wp.launch(
                            kernel,
                            1,
                            [
                                world._contact_cols,
                                world.bodies,
                                world.particles or ParticleContainer(),
                                world.num_bodies,
                                cc,
                                world._active_contact_views(),
                                copies,
                            ],
                            device=model.device,
                        )
                        results.append(
                            (
                                world.bodies.velocity.numpy(),
                                world.bodies.angular_velocity.numpy(),
                                copies.velocity.numpy(),
                                copies.angular_velocity.numpy(),
                                cc.impulses.numpy(),
                                cc.lambdas.numpy(),
                                cc.derived.numpy(),
                                world._contact_cols.data.numpy(),
                            )
                        )
                    for expected, actual in zip(*results, strict=True):
                        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


if __name__ == "__main__":
    unittest.main()
