# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check physical static solves independently of solver dispatch overlays."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, MOTION_STATIC, body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_column_container_zeros,
    contact_set_body1,
    contact_set_body2,
    contact_set_contact_count,
    contact_set_contact_first,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_set_eff_n,
    cc_set_normal,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_tgs import ContactTGS, NormalRow, allocate_contact_tgs
from newton._src.solvers.phoenx.constraints.contact_tgs_static import build_lists, get_sweep
from newton._src.solvers.phoenx.mass_splitting.copy_state import copy_state_container_zeros


@wp.kernel
def setup(
    columns: ContactColumnContainer,
    cc: ContactContainer,
    state: ContactTGS,
    pairs: wp.array[wp.vec2i],
    response: wp.array[float],
):
    cid = wp.tid()
    contact_set_body1(columns, cid, pairs[cid][0])
    contact_set_body2(columns, cid, pairs[cid][1])
    contact_set_contact_first(columns, cid, cid)
    contact_set_contact_count(columns, cid, 1)
    row = NormalRow()
    row.normal = wp.vec3f(0.0, 0.0, 1.0)
    row.effective_mass = response[cid]
    state.normal_rows[cid] = row
    cc_set_normal(cc, cid, row.normal)
    cc_set_eff_n(cc, cid, row.effective_mass)


class TestContactTGSStatic(unittest.TestCase):
    def test_physical_solve_and_copy_broadcast(self):
        """Solve each static column once and broadcast physical velocities to all copies."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                self._check_device(device)

    def _check_device(self, device):
        bodies = body_container_zeros(4, device)
        bodies.motion_type.assign([MOTION_DYNAMIC, MOTION_DYNAMIC, MOTION_DYNAMIC, MOTION_STATIC])
        bodies.inverse_mass.assign([1.0, 0.5, 2.0, 0.0])
        bodies.orientation.assign(np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (4, 1)))
        bodies.velocity.assign([[0, 0, 9], [0, 0, 9], [0, 0, 3], [0, 0, 0]])
        columns = contact_column_container_zeros(5, device)
        cc = contact_container_zeros(5, device)
        state = allocate_contact_tgs(5, 4, 3, device)
        pairs = wp.array([[0, 3], [1, 2], [3, 1], [0, 3], [2, 3]], dtype=wp.vec2i, device=device)
        response = wp.array([1, 0.4, 2, 1, 0.5], dtype=float, device=device)
        wp.launch(setup, 5, [columns, cc, state, pairs, response], device=device)
        active = wp.array([5], dtype=int, device=device)
        heads = wp.full(4, -1, dtype=int, device=device)
        links = wp.full(5, -1, dtype=int, device=device)
        wp.launch(build_lists, 4, [columns, bodies, active, heads, links], device=device)
        np.testing.assert_array_equal(heads.numpy(), [0, 2, 4, -1])
        np.testing.assert_array_equal(links.numpy(), [3, -1, -1, -1, -1])
        copies = copy_state_container_zeros(3, 4, device)
        copies.count_per_node.assign([2, 1, 0, 0])
        copies.section_end.assign([2, 3, 3, 3])
        copies.velocity.assign([[0, 0, 1], [0, 0, 1], [0, 0, -2]])
        args = [columns, state, bodies, cc, copies, heads, links, 100.0]
        wp.launch(get_sweep("prepare"), 4, args, device=device)
        np.testing.assert_array_equal(bodies.velocity.numpy()[:, 2], [1, -2, 3, 0])
        np.testing.assert_array_equal(cc.impulses.numpy(), 0)
        for phase in ("iterate", "relax"):
            wp.launch(get_sweep(phase), 4, args, device=device)
            np.testing.assert_array_equal(bodies.velocity.numpy(), 0)
            np.testing.assert_array_equal(copies.velocity.numpy(), 0)
            np.testing.assert_array_equal(bodies.angular_velocity.numpy(), 0)
            np.testing.assert_array_equal(copies.angular_velocity.numpy(), 0)
            # Opposite endpoint order reverses the external impulse sign.
            np.testing.assert_array_equal(cc.impulses.numpy()[0], [1, 0, 4, 0, 1.5])
        # Rebuild after contacts disappear: no stale static work may survive.
        active.assign([0])
        wp.launch(build_lists, 4, [columns, bodies, active, heads, links], device=device)
        np.testing.assert_array_equal(heads.numpy(), -1)


if __name__ == "__main__":
    unittest.main()
