# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Graph-coloring integration test for cloth-triangle constraints.

Commit 4 of :file:`PLAN_CLOTH_TRIANGLE.md`. Verifies:

* The cloth-aware ``_constraints_to_elements_cloth_kernel`` reads
  all three cloth-triangle endpoints (``body1``, ``body2``,
  ``body3``) into the per-cid
  :class:`ElementInteractionData`. The base kernel (no cloth)
  would only emit two endpoints, missing the third particle for
  the colourer.
* Static-body collapse uses the unified
  :func:`get_inverse_mass` so cloth particles with
  ``inverse_mass = 0`` (pinned corners) get masked out the same
  way static rigid bodies do.

This test runs against the kernel directly, not through the full
PhoenXWorld solve, so it focuses on the data-flow correctness of
the new cloth-aware variant.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.body_or_particle import BodyOrParticleStore
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    CLOTH_TRIANGLE_DWORDS,
    cloth_triangle_set_body1,
    cloth_triangle_set_body2,
    cloth_triangle_set_body3,
    cloth_triangle_set_type,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_container_zeros,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    ElementInteractionData,
)
from newton._src.solvers.phoenx.particle import particle_container_zeros
from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    _constraints_to_elements_cloth_kernel,
)


@unittest.skipUnless(wp.is_cuda_available(), "Cloth-coloring tests require CUDA")
class TestClothElementInteraction(unittest.TestCase):
    """The cloth-aware constraints-to-elements kernel must emit
    three endpoints for cloth cids (vs. two for ADBS / contact)."""

    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_cloth_emits_three_endpoints(self) -> None:
        # 1 cloth triangle with three particle endpoints. Bodies
        # array has just slot 0 (anchor); particles 0..2 sit at
        # unified indices 1..3. None static, so all three should
        # emerge from the kernel.
        num_bodies = 1
        num_particles = 3
        num_constraints = 1

        constraints = constraint_container_zeros(
            num_constraints=num_constraints,
            num_dwords=CLOTH_TRIANGLE_DWORDS,
            device=self.device,
        )
        # Stamp one cloth triangle row.
        wp.launch(
            kernel=_stamp_cloth_row_kernel,
            dim=1,
            inputs=[constraints, wp.int32(1), wp.int32(2), wp.int32(3)],
            device=self.device,
        )

        bodies = body_container_zeros(num_bodies, device=self.device)
        # Anchor bodies slot 0 has zero mass; particles all dynamic.
        particles = particle_container_zeros(num_particles, device=self.device)
        particles.inverse_mass.assign(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        store = BodyOrParticleStore()
        store.bodies = bodies
        store.particles = particles
        store.num_bodies = wp.int32(num_bodies)

        # Sentinel contact column container (no contacts in this
        # test).
        from newton._src.solvers.phoenx.constraints.constraint_contact import (
            contact_column_container_zeros,
        )

        contact_cols = contact_column_container_zeros(num_columns=1, device=self.device)
        elements = wp.zeros(num_constraints, dtype=ElementInteractionData, device=self.device)
        num_active = wp.array([num_constraints], dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=_constraints_to_elements_cloth_kernel,
            dim=num_constraints,
            inputs=[
                constraints,
                contact_cols,
                bodies,
                num_active,
                wp.int32(num_constraints),  # num_joints == num_constraints (no contacts)
                elements,
                store,
            ],
            device=self.device,
        )

        out = elements.numpy()
        # ElementInteractionData.bodies is an 8-slot vec8i; for our
        # 3-particle cloth, slots 0..2 should be the unified
        # indices and slots 3..7 should be -1.
        bodies_for_cid0 = out[0]["bodies"]
        # Sort to compare set-wise (order is "non-negative first" so
        # exact order is deterministic but we don't pin which
        # particle ends up first).
        first_three = sorted(int(x) for x in bodies_for_cid0[:3])
        rest = [int(x) for x in bodies_for_cid0[3:]]
        self.assertEqual(first_three, [1, 2, 3])
        self.assertEqual(rest, [-1, -1, -1, -1, -1])

    def test_pinned_corner_collapsed_to_minus_one(self) -> None:
        """A particle with ``inverse_mass = 0`` is treated as static
        by the colourer (collapsed to ``-1``). Cloth pinning ride
        on this convention -- pinned corners shouldn't conflict
        with anything in the colouring graph."""
        num_bodies = 1
        num_particles = 3
        num_constraints = 1

        constraints = constraint_container_zeros(
            num_constraints=num_constraints,
            num_dwords=CLOTH_TRIANGLE_DWORDS,
            device=self.device,
        )
        wp.launch(
            kernel=_stamp_cloth_row_kernel,
            dim=1,
            inputs=[constraints, wp.int32(1), wp.int32(2), wp.int32(3)],
            device=self.device,
        )
        bodies = body_container_zeros(num_bodies, device=self.device)
        particles = particle_container_zeros(num_particles, device=self.device)
        # Pin particle 0 (unified index 1) by setting its
        # inverse_mass to 0; the other two are dynamic.
        particles.inverse_mass.assign(np.array([0.0, 1.0, 1.0], dtype=np.float32))
        store = BodyOrParticleStore()
        store.bodies = bodies
        store.particles = particles
        store.num_bodies = wp.int32(num_bodies)
        from newton._src.solvers.phoenx.constraints.constraint_contact import (
            contact_column_container_zeros,
        )

        contact_cols = contact_column_container_zeros(num_columns=1, device=self.device)
        elements = wp.zeros(num_constraints, dtype=ElementInteractionData, device=self.device)
        num_active = wp.array([num_constraints], dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=_constraints_to_elements_cloth_kernel,
            dim=num_constraints,
            inputs=[
                constraints,
                contact_cols,
                bodies,
                num_active,
                wp.int32(num_constraints),
                elements,
                store,
            ],
            device=self.device,
        )
        out = elements.numpy()
        bodies_for_cid0 = out[0]["bodies"]
        # Two non-negative endpoints (the dynamic particles 2 and
        # 3 -- unified indices), then -1 padding.
        first_two = sorted(int(x) for x in bodies_for_cid0[:2])
        rest = [int(x) for x in bodies_for_cid0[2:]]
        self.assertEqual(first_two, [2, 3])
        # All remaining slots must be -1 (the pinned corner got
        # collapsed and the cloth doesn't have a 4th endpoint).
        self.assertTrue(all(x == -1 for x in rest))


# ---------------------------------------------------------------------------
# Test-only helper kernels.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _stamp_cloth_row_kernel(
    c: ConstraintContainer,
    body_a: wp.int32,
    body_b: wp.int32,
    body_c: wp.int32,
):
    cid = wp.tid()
    cloth_triangle_set_type(c, cid)
    cloth_triangle_set_body1(c, cid, body_a)
    cloth_triangle_set_body2(c, cid, body_b)
    cloth_triangle_set_body3(c, cid, body_c)


if __name__ == "__main__":
    unittest.main()
