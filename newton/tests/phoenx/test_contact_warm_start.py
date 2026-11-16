# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare canonical gathered warm starts with the colored contact update."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer, body_container_zeros, inertia_sym6
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_column_container_zeros,
    contact_owned_elsewhere,
    contact_set_body1,
    contact_set_body2,
    contact_set_contact_count,
    contact_set_contact_first,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_solve import contact_cached_warmstart_lean
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_set_normal,
    cc_set_normal_lambda,
    cc_set_r0,
    cc_set_r1,
    cc_set_tangent1,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_warm_start import ContactBodyEntries
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer


@wp.kernel
def _initialize_columns(columns: ContactColumnContainer, cc: ContactContainer):
    cid = wp.tid()
    contact_set_body1(columns, cid, cid % 4)
    contact_set_body2(columns, cid, 1 + (cid + 1) % 4)
    contact_set_contact_first(columns, cid, 2 * cid)
    contact_set_contact_count(columns, cid, 2)
    if cid == 4:
        columns.articulation_owner[cid] = 0
    if cid == 5:
        columns.static_owner[cid] = 1
    for j in range(2):
        k = 2 * cid + j
        cc_set_normal(cc, k, wp.vec3(0.0, 0.0, 1.0))
        cc_set_tangent1(cc, k, wp.vec3(1.0, 0.0, 0.0))
        cc_set_r0(cc, k, wp.vec3(0.1 * wp.float32(j + 1), -0.2, 0.3))
        cc_set_r1(cc, k, wp.vec3(-0.3, 0.2 * wp.float32(j + 1), -0.1))
        cc_set_normal_lambda(cc, k, 0.3 * wp.float32(k + 1))
        cc_set_tangent1_lambda(cc, k, -0.1 * wp.float32(k + 1))
        cc_set_tangent2_lambda(cc, k, 0.07 * wp.float32(k + 1))


@wp.kernel
def _colored_reference(
    columns: ContactColumnContainer,
    cc: ContactContainer,
    bodies: BodyContainer,
    ids: wp.array[wp.int32],
    active: wp.array[wp.int32],
):
    for slot in range(active[0]):
        cid = ids[slot] - 2
        if cid >= 0:
            if not contact_owned_elsewhere(columns, cid):
                contact_cached_warmstart_lean(
                    columns, cid, bodies, ParticleContainer(), 5, 120.0, cc, ContactViews(), CopyStateContainer(), 0
                )


class TestContactWarmStart(unittest.TestCase):
    @staticmethod
    def _fixture():
        columns = contact_column_container_zeros(7)
        cc = contact_container_zeros(14)
        wp.launch(_initialize_columns, 7, inputs=[columns, cc])
        # Joint slots, shared bodies, two excluded owners, and one inactive column.
        ids = wp.array([4, 0, 2, 7, 3, 1, 6, 5, 8], dtype=wp.int32)
        active = wp.array([8], dtype=wp.int32)
        entries = ContactBodyEntries(9, 14, 5, wp.get_device())
        entries.build(columns, ids, active, 2, columns.articulation_owner, packed_columns=False)
        return columns, cc, ids, active, entries

    def test_canonical_entries_preserve_order_and_reset(self):
        """Exclude owned and inactive columns while retaining endpoint order."""
        with wp.ScopedDevice("cpu"):
            columns, _, ids, active, entries = self._fixture()
            starts, values = entries.starts.numpy(), entries.values.numpy()
            expected = ([0], [2, 7], [4, 1], [3, 6], [5])
            for body, codes in enumerate(expected):
                np.testing.assert_array_equal(values[starts[body] : starts[body + 1]], codes)
            active.zero_()
            entries.build(columns, ids, active, 2, columns.articulation_owner, packed_columns=False)
            np.testing.assert_array_equal(entries.starts.numpy(), 0)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA subgroup gather required")
    def test_cached_gather_matches_colored_impulses_cuda(self):
        """Replay normal and friction impulses without changing their values."""
        with wp.ScopedDevice("cuda:0"):
            columns, cc, ids, active, entries = self._fixture()
            bodies = body_container_zeros(5)
            bodies.inverse_mass.assign(np.array([0, 0.2, 3.0, 0.5, 0], dtype=np.float32))
            rng = np.random.default_rng(42)
            original_impulses = cc.impulses.numpy()
            for trial in range(4):
                inertia = rng.uniform(0.2, 3.0, (5, 6)).astype(np.float32)
                # Symmetric storage is xx, xy, xz, yy, yz, zz. Keep the
                # off-diagonal entries small enough for positive definiteness.
                inertia[:, [1, 2, 4]] *= 0.01
                inertia[[0, 4]] = 0
                wp.copy(bodies.inverse_inertia_world, wp.array(inertia, dtype=inertia_sym6))
                velocity = rng.normal(size=bodies.velocity.numpy().shape).astype(np.float32)
                angular = rng.normal(size=bodies.angular_velocity.numpy().shape).astype(np.float32)
                cc.impulses.assign(original_impulses * (trial + 1))
                before = cc.impulses.numpy()
                bodies.velocity.assign(velocity)
                bodies.angular_velocity.assign(angular)
                wp.launch(_colored_reference, 1, inputs=[columns, cc, bodies, ids, active])
                expected_v, expected_w = bodies.velocity.numpy(), bodies.angular_velocity.numpy()
                bodies.velocity.assign(velocity)
                bodies.angular_velocity.assign(angular)
                entries.apply_cached_warm_start(columns, bodies, cc, ids, active, 2)
                np.testing.assert_allclose(bodies.velocity.numpy(), expected_v, atol=2e-6, rtol=2e-6)
                np.testing.assert_allclose(bodies.angular_velocity.numpy(), expected_w, atol=2e-6, rtol=2e-6)
                np.testing.assert_array_equal(cc.impulses.numpy(), before)
            active.zero_()
            entries.build(columns, ids, active, 2, columns.articulation_owner, packed_columns=False)
            before_v, before_w = bodies.velocity.numpy(), bodies.angular_velocity.numpy()
            entries.apply_cached_warm_start(columns, bodies, cc, ids, active, 2)
            np.testing.assert_array_equal(bodies.velocity.numpy(), before_v)
            np.testing.assert_array_equal(bodies.angular_velocity.numpy(), before_w)


if __name__ == "__main__":
    unittest.main()
