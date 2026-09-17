# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify physical momentum after temporal contact body-copy dispatch."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer, body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_column_container_zeros,
    contact_set_body1,
    contact_set_body2,
    contact_set_contact_count,
    contact_set_contact_first,
    contact_set_count1,
    contact_set_count2,
    contact_set_slot1,
    contact_set_slot2,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_set_eff_n,
    cc_set_normal,
    cc_set_r0,
    cc_set_r1,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_tgs import ContactTGS, NormalRow, allocate_contact_tgs
from newton._src.solvers.phoenx.constraints.contact_tgs_dynamic import make_iterate
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer, copy_state_container_zeros


@wp.kernel
def setup(
    columns: ContactColumnContainer,
    cc: ContactContainer,
    state: ContactTGS,
    slot0: int,
    slot1: int,
    count0: int,
    count1: int,
    effective_mass: float,
    size: int,
):
    contact_set_body1(columns, 0, 0)
    contact_set_body2(columns, 0, 1)
    contact_set_contact_first(columns, 0, 0)
    contact_set_contact_count(columns, 0, size)
    contact_set_count1(columns, 0, count0)
    contact_set_count2(columns, 0, count1)
    contact_set_slot1(columns, 0, slot0)
    contact_set_slot2(columns, 0, slot1)
    row = NormalRow()
    row.normal = wp.vec3f(0.0, 0.0, 1.0)
    row.r0 = wp.vec3f(1.0, 0.0, 0.0)
    row.r1 = wp.vec3f(1.0, 0.0, -1.0)
    row.effective_mass = effective_mass
    for k in range(size):
        state.normal_rows[k] = row
        cc_set_normal(cc, k, row.normal)
        cc_set_eff_n(cc, k, row.effective_mass)
        cc_set_r0(cc, k, row.r0)
        cc_set_r1(cc, k, row.r1)


def kernel(mass_splitting, biased, cooperative=False, record_wrenches=False):
    iterate = make_iterate(
        mass_splitting=mass_splitting, biased=biased, cooperative=cooperative, record_wrenches=record_wrenches
    )

    @wp.kernel(module="unique")
    def run(
        columns: ContactColumnContainer,
        state: ContactTGS,
        bodies: BodyContainer,
        cc: ContactContainer,
        copies: CopyStateContainer,
    ):
        iterate(columns, state, 0, bodies, cc, copies, 100.0, wp.tid())

    return run


class TestContactTGSDynamic(unittest.TestCase):
    def test_copy_response_preserves_physical_momenta(self):
        """Conserve both physical momenta after averaging split and unsplit endpoints."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            for split, slot0, slot1, count0, count1 in ((True, 0, 3, 3, 2), (True, -1, 3, 1, 2), (False, -1, -1, 1, 1)):
                for biased in (False, True):
                    with self.subTest(device=device, split=split, slot0=slot0, biased=biased):
                        self._check_case(device, split, biased, slot0, slot1, count0, count1)

    def test_cooperative_copy_response_preserves_physical_momenta(self):
        """Preserve physical momentum across full and partial CUDA subgroups."""
        if not wp.is_cuda_available():
            self.skipTest("Cooperative row loading requires CUDA")
        for size in (1, 7, 8, 9, 33):
            for biased in (False, True):
                with self.subTest(size=size, biased=biased):
                    self._check_case("cuda:0", True, biased, 0, 3, 3, 2, True, size)

    def test_recorded_wrench_matches_copy_response(self):
        """Match physical impulse and world moment in scalar and cooperative solves."""
        for cooperative in (False, True):
            if cooperative and not wp.is_cuda_available():
                continue
            device = "cuda:0" if cooperative else "cpu"
            for biased in (False, True):
                self._check_case(device, True, biased, 0, 3, 3, 2, cooperative, 9, True)

    def _check_case(
        self, device, split, biased, slot0, slot1, count0, count1, cooperative=False, size=1, record_wrenches=False
    ):
        bodies = body_container_zeros(2, device)
        bodies.inverse_mass.assign([1.0, 0.5])
        bodies.position.assign([[0, 0, 0], [0, 0, 1]])
        bodies.orientation.assign([[0, 0, 0, 1], [0, 0, 0, 1]])
        bodies.inverse_inertia_world.assign([[0.5, 0.5, 0.5, 0, 0, 0], [0.25, 0.25, 0.25, 0, 0, 0]])
        initial = np.array([[0, 0, 1], [0, 0, -1]], dtype=np.float32)
        bodies.velocity.assign(initial)
        copies = copy_state_container_zeros(5, 2, device)
        initial_copies = initial[[0, 0, 0, 1, 1]]
        copies.velocity.assign(initial_copies)
        columns = contact_column_container_zeros(1, device)
        cc = contact_container_zeros(size, device)
        state = allocate_contact_tgs(size, 2, 3, device)
        state.record_wrenches = int(record_wrenches)
        eff = 1.0 / (1.5 * count0 + 0.75 * count1)
        wp.launch(setup, 1, [columns, cc, state, slot0, slot1, count0, count1, eff, size], device=device)
        wp.launch(
            kernel(split, biased, cooperative, record_wrenches),
            8 if cooperative else 1,
            [columns, state, bodies, cc, copies],
            device=device,
        )
        v = bodies.velocity.numpy().astype(float)
        w = bodies.angular_velocity.numpy().astype(float)
        cv = copies.velocity.numpy().astype(float)
        cw = copies.angular_velocity.numpy().astype(float)
        for body, slot, count in ((0, slot0, count0), (1, slot1, count1)):
            if slot >= 0:
                np.testing.assert_array_equal(v[body], initial[body])
                v[body] = cv[slot : slot + count].mean(axis=0)
                w[body] = cw[slot : slot + count].mean(axis=0)
        np.testing.assert_allclose(v[0] + 2 * v[1], initial[0] + 2 * initial[1], atol=2e-7)
        angular = 2 * w[0] + 4 * w[1] + np.cross([0, 0, 1], 2 * v[1])
        np.testing.assert_allclose(angular, 0, atol=2e-7)
        np.testing.assert_allclose(cc.impulses.numpy()[0, 0], 2 * eff, rtol=1e-6)
        if record_wrenches:
            wrench = state.wrenches.numpy().sum(axis=0)
            np.testing.assert_allclose(wrench[:3], 2 * (v[1] - initial[1]), atol=2e-7)
            np.testing.assert_allclose(wrench[3:], 4 * w[1] + np.cross([0, 0, 1], wrench[:3]), atol=2e-7)
        for slot in range(5):
            if slot not in (slot0, slot1):
                np.testing.assert_array_equal(cv[slot], initial_copies[slot])
                np.testing.assert_array_equal(cw[slot], 0)


if __name__ == "__main__":
    unittest.main()
