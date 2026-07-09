# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA-graph tests for general color-ordered contact-row staging."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact import (
    RIGID_CONTACT_SOLVE_DWORDS,
    ContactColumnContainer,
    contact_build_colored_row_offsets,
    contact_column_container_zeros,
    contact_gather_colored_rows,
    contact_pack_colored_headers,
    contact_scatter_colored_rows,
    contact_set_contact_count,
    contact_set_contact_first,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    contact_container_zeros,
    contact_solve_container_zeros,
)
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene


@wp.kernel(enable_backward=False)
def _set_contact_ranges(
    headers: ContactColumnContainer,
    first: wp.array[wp.int32],
    count: wp.array[wp.int32],
):
    cid = wp.tid()
    contact_set_contact_first(headers, cid, first[cid])
    contact_set_contact_count(headers, cid, count[cid])


@wp.kernel(enable_backward=False)
def _offset_packed_impulses(contacts: ContactContainer, total: wp.array[wp.int32]):
    k = wp.tid()
    if k >= total[0]:
        return
    for row in range(3):
        contacts.impulses[row, k] += wp.float32(10000.0)


@unittest.skipUnless(wp.is_cuda_available(), "colored contact-row tests require CUDA graph capture")
class TestColoredContactRows(unittest.TestCase):
    def test_zero_velocity_iterations_restores_canonical_rows(self) -> None:
        """The solve-only scatter path runs when the relax phase is disabled."""
        scene = _PhoenXScene(
            fps=60,
            substeps=4,
            solver_iterations=3,
            velocity_iterations=0,
            step_layout="single_world",
            mass_splitting=True,
            colored_contact_headers=True,
            colored_contact_rows=True,
            max_colored_partitions=8,
            mass_splitting_batch_size=1,
        )
        scene.add_ground_plane()
        scene.add_box((0.0, 0.0, 0.55), (0.5, 0.5, 0.5))
        scene.add_box((0.0, 0.0, 1.55), (0.5, 0.5, 0.5))
        scene.finalize()
        for _ in range(30):
            scene.step()

        self.assertIsNotNone(scene._graph)
        self.assertTrue(np.isfinite(scene.bodies.position.numpy()).all())
        self.assertTrue(np.isfinite(scene.bodies.velocity.numpy()).all())

    def test_arbitrary_contact_counts_round_trip_in_graph(self) -> None:
        """Interleaved columns with 1, 7, and 13 contacts have no row cap."""
        device = wp.get_device("cuda:0")
        source_headers = contact_column_container_zeros(3, device=device)
        packed_headers = contact_column_container_zeros(
            4,
            device=device,
            data_dwords=RIGID_CONTACT_SOLVE_DWORDS,
        )
        first = wp.array(np.array([0, 7, 8], dtype=np.int32), dtype=wp.int32, device=device)
        count = wp.array(np.array([7, 1, 13], dtype=np.int32), dtype=wp.int32, device=device)
        wp.launch(_set_contact_ranges, dim=3, inputs=[source_headers, first, count], device=device)

        row_capacity = 24
        source = contact_container_zeros(row_capacity, device=device)
        packed = contact_solve_container_zeros(row_capacity, device=device)
        impulse_np = np.arange(3 * row_capacity, dtype=np.float32).reshape(3, row_capacity)
        lambda_np = (1000.0 + np.arange(18 * row_capacity, dtype=np.float32)).reshape(18, row_capacity)
        derived_np = (2000.0 + np.arange(16 * row_capacity, dtype=np.float32)).reshape(16, row_capacity)
        source.impulses.assign(impulse_np)
        source.lambdas.assign(lambda_np)
        source.derived.assign(derived_np)

        # Color slots: source column 1, a non-contact joint, then columns 0 and 2.
        element_ids = wp.array(np.array([101, 5, 100, 102], dtype=np.int32), dtype=wp.int32, device=device)
        num_active = wp.array(np.array([4], dtype=np.int32), dtype=wp.int32, device=device)
        counts = wp.zeros(4, dtype=wp.int32, device=device)
        offsets = wp.zeros(4, dtype=wp.int32, device=device)
        source_indices = wp.zeros(row_capacity, dtype=wp.int32, device=device)
        slots = wp.zeros(row_capacity, dtype=wp.int32, device=device)
        total = wp.zeros(1, dtype=wp.int32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            contact_pack_colored_headers(
                source_headers,
                packed_headers,
                element_ids,
                num_active,
                contact_offset=100,
                capacity=4,
                device=device,
            )
            contact_build_colored_row_offsets(
                packed_headers,
                element_ids,
                num_active,
                contact_offset=100,
                counts=counts,
                offsets=offsets,
                source_indices=source_indices,
                slots=slots,
                total=total,
                capacity=4,
                device=device,
            )
            contact_gather_colored_rows(
                source,
                packed,
                packed_headers,
                offsets,
                source_indices,
                slots,
                total,
                row_capacity,
                device=device,
            )
            wp.launch(_offset_packed_impulses, dim=row_capacity, inputs=[packed, total], device=device)
            contact_scatter_colored_rows(
                packed,
                source,
                packed_headers,
                offsets,
                source_indices,
                slots,
                total,
                row_capacity,
                device=device,
            )
        wp.capture_launch(capture.graph)

        expected_order = np.array([7, *range(7), *range(8, 21)], dtype=np.int32)
        np.testing.assert_array_equal(total.numpy(), np.array([21], dtype=np.int32))
        np.testing.assert_array_equal(source_indices.numpy()[:21], expected_order)
        np.testing.assert_array_equal(offsets.numpy(), np.array([0, 1, 1, 8], dtype=np.int32))
        np.testing.assert_array_equal(packed.lambdas.numpy()[:, :21], lambda_np[:6, expected_order])
        np.testing.assert_array_equal(packed.derived.numpy()[:, :21], derived_np[:15, expected_order])

        expected_impulses = impulse_np.copy()
        expected_impulses[:, :21] += 10000.0
        np.testing.assert_array_equal(source.impulses.numpy(), expected_impulses)


if __name__ == "__main__":
    unittest.main()
