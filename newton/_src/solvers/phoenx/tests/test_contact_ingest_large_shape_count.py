# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression test for ``Bug.md`` #1: int32 overflow in contact-pair key packing.

The pre-fix ingest kernel packed ``(shape_a, shape_b)`` into a single int32
via ``sa * num_shapes + sb``. When ``num_shapes >= 46_341`` the product
silently wrapped, which routed every contact constraint through a single
aliased body slot and wiped out static-ground handling -- h1_flat @
900+ worlds was the observed crash.

The fix replaced the key packing with an adjacency-based run detector:
since contacts are already sorted by ``(shape_a, shape_b)``, we just
compare the current entry against the previous one to mark a new run,
inclusive-scan into run ids, and scatter pair metadata at the boundary
positions. No key packing, no int32 overflow.

This test exercises the ingest kernels directly with synthetic contact
arrays containing shape ids > 46_340, confirming that
``pair_shape_a``, ``pair_shape_b``, ``pair_first``, and ``pair_count``
come out correctly -- which they did not under the old code path.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_ingest import (
    _contact_pair_boundary_kernel,
    _pair_counts_from_starts_kernel,
    _scatter_pair_starts_kernel,
)


def _synthesize_sorted_contacts(
    pairs: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Expand ``[(shape_a, shape_b, run_length)]`` into a sorted per-contact
    ``(shape0, shape1)`` stream.

    Returns ``(shape0, shape1, count)`` as numpy arrays + the total count.
    Useful because the fix is driven entirely off already-sorted
    ``(shape0, shape1)`` -- we don't need real contact geometry here.
    """
    shape0_list: list[int] = []
    shape1_list: list[int] = []
    for sa, sb, rl in pairs:
        shape0_list.extend([sa] * rl)
        shape1_list.extend([sb] * rl)
    return (
        np.asarray(shape0_list, dtype=np.int32),
        np.asarray(shape1_list, dtype=np.int32),
        len(shape0_list),
    )


def _run_pair_detection(
    shape0_np: np.ndarray,
    shape1_np: np.ndarray,
    count: int,
    capacity: int,
    device,
) -> dict:
    """Run the three pair-detection kernels + the scan on host-supplied
    synthetic contact data. Returns the post-kernel device buffers as
    numpy arrays."""
    assert capacity >= count
    # Pad contact arrays to ``capacity`` with garbage past ``count`` so
    # the test also verifies the tail-ignore behaviour.
    shape0_pad = np.full(capacity, -1, dtype=np.int32)
    shape1_pad = np.full(capacity, -1, dtype=np.int32)
    shape0_pad[:count] = shape0_np
    shape1_pad[:count] = shape1_np

    shape0 = wp.from_numpy(shape0_pad, dtype=wp.int32, device=device)
    shape1 = wp.from_numpy(shape1_pad, dtype=wp.int32, device=device)
    count_arr = wp.array([count], dtype=wp.int32, device=device)

    pair_boundary = wp.zeros(capacity, dtype=wp.int32, device=device)
    pair_id = wp.zeros(capacity, dtype=wp.int32, device=device)
    pair_shape_a = wp.zeros(capacity, dtype=wp.int32, device=device)
    pair_shape_b = wp.zeros(capacity, dtype=wp.int32, device=device)
    pair_first = wp.zeros(capacity, dtype=wp.int32, device=device)
    pair_columns = wp.zeros(capacity, dtype=wp.int32, device=device)
    pair_count = wp.zeros(capacity, dtype=wp.int32, device=device)
    num_pairs = wp.zeros(1, dtype=wp.int32, device=device)

    wp.launch(
        _contact_pair_boundary_kernel,
        dim=capacity,
        inputs=[count_arr, shape0, shape1],
        outputs=[pair_boundary],
        device=device,
    )
    wp.utils.array_scan(pair_boundary, pair_id, inclusive=True)
    wp.launch(
        _scatter_pair_starts_kernel,
        dim=capacity,
        inputs=[count_arr, pair_boundary, pair_id, shape0, shape1],
        outputs=[pair_shape_a, pair_shape_b, pair_first, pair_columns, num_pairs],
        device=device,
    )
    wp.launch(
        _pair_counts_from_starts_kernel,
        dim=capacity,
        inputs=[count_arr, num_pairs, pair_first],
        outputs=[pair_count],
        device=device,
    )

    n = int(num_pairs.numpy()[0])
    return {
        "num_pairs": n,
        "pair_shape_a": pair_shape_a.numpy()[:n],
        "pair_shape_b": pair_shape_b.numpy()[:n],
        "pair_first": pair_first.numpy()[:n],
        "pair_count": pair_count.numpy()[:n],
        "pair_boundary": pair_boundary.numpy()[:count],
        "pair_id": pair_id.numpy()[:count],
    }


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX ingest regression tests require CUDA")
class TestContactIngestLargeShapeCount(unittest.TestCase):
    """Regression: pair detection must stay correct above the old int32
    key overflow threshold."""

    def test_pairs_with_shape_ids_above_int32_overflow_threshold(self) -> None:
        """The old packing ``sa * num_shapes + sb`` overflowed for
        ``sa * num_shapes >= 2**31`` (~ 46_340 shapes at equal sa/sb).

        This test uses shape ids well above that point to guarantee
        that *any* int32 packing would wrap, and verifies the
        adjacency-mark pipeline still produces the right pair layout.
        """
        pairs = [
            # (shape_a, shape_b, contact_count) -- sorted by (sa, sb).
            (50_000, 50_001, 2),
            (50_000, 50_002, 3),
            (50_100, 50_001, 1),
            (50_100, 50_200, 4),
            (60_000, 60_001, 5),
        ]
        shape0, shape1, count = _synthesize_sorted_contacts(pairs)
        device = wp.get_preferred_device()
        result = _run_pair_detection(shape0, shape1, count, capacity=32, device=device)

        self.assertEqual(result["num_pairs"], len(pairs))
        expected_sa = np.array([p[0] for p in pairs], dtype=np.int32)
        expected_sb = np.array([p[1] for p in pairs], dtype=np.int32)
        expected_counts = np.array([p[2] for p in pairs], dtype=np.int32)
        expected_first = np.concatenate(([0], np.cumsum(expected_counts[:-1]))).astype(np.int32)

        np.testing.assert_array_equal(result["pair_shape_a"], expected_sa)
        np.testing.assert_array_equal(result["pair_shape_b"], expected_sb)
        np.testing.assert_array_equal(result["pair_first"], expected_first)
        np.testing.assert_array_equal(result["pair_count"], expected_counts)

    def test_single_pair_single_contact(self) -> None:
        """Edge case: one contact for one pair. Pair 0 runs [0, 1)."""
        shape0, shape1, count = _synthesize_sorted_contacts([(100_000, 200_000, 1)])
        device = wp.get_preferred_device()
        result = _run_pair_detection(shape0, shape1, count, capacity=4, device=device)
        self.assertEqual(result["num_pairs"], 1)
        self.assertEqual(int(result["pair_shape_a"][0]), 100_000)
        self.assertEqual(int(result["pair_shape_b"][0]), 200_000)
        self.assertEqual(int(result["pair_first"][0]), 0)
        self.assertEqual(int(result["pair_count"][0]), 1)

    def test_empty_contact_array(self) -> None:
        """Zero active contacts -> zero pairs, everything stable."""
        device = wp.get_preferred_device()
        empty = np.empty(0, dtype=np.int32)
        result = _run_pair_detection(empty, empty, 0, capacity=4, device=device)
        self.assertEqual(result["num_pairs"], 0)

    def test_stale_tail_past_active_length_ignored(self) -> None:
        """The tail past ``count`` must not affect pair detection --
        the boundary kernel zeros those slots, the scan's inclusive
        tail does not leak boundaries, and scatter only fires for
        ``tid < count``."""
        shape0, shape1, count = _synthesize_sorted_contacts([(1, 2, 3), (4, 5, 2)])
        # Put conflicting garbage past the active count.
        device = wp.get_preferred_device()
        capacity = 16
        shape0_pad = np.arange(capacity, dtype=np.int32) * 97
        shape1_pad = np.arange(capacity, dtype=np.int32) * 131
        shape0_pad[:count] = shape0
        shape1_pad[:count] = shape1

        count_arr = wp.array([count], dtype=wp.int32, device=device)
        shape0_arr = wp.from_numpy(shape0_pad, dtype=wp.int32, device=device)
        shape1_arr = wp.from_numpy(shape1_pad, dtype=wp.int32, device=device)
        pair_boundary = wp.zeros(capacity, dtype=wp.int32, device=device)
        pair_id = wp.zeros(capacity, dtype=wp.int32, device=device)
        pair_shape_a = wp.zeros(capacity, dtype=wp.int32, device=device)
        pair_shape_b = wp.zeros(capacity, dtype=wp.int32, device=device)
        pair_first = wp.zeros(capacity, dtype=wp.int32, device=device)
        pair_columns = wp.zeros(capacity, dtype=wp.int32, device=device)
        pair_count = wp.zeros(capacity, dtype=wp.int32, device=device)
        num_pairs = wp.zeros(1, dtype=wp.int32, device=device)

        wp.launch(
            _contact_pair_boundary_kernel,
            dim=capacity,
            inputs=[count_arr, shape0_arr, shape1_arr],
            outputs=[pair_boundary],
            device=device,
        )
        wp.utils.array_scan(pair_boundary, pair_id, inclusive=True)
        wp.launch(
            _scatter_pair_starts_kernel,
            dim=capacity,
            inputs=[count_arr, pair_boundary, pair_id, shape0_arr, shape1_arr],
            outputs=[pair_shape_a, pair_shape_b, pair_first, pair_columns, num_pairs],
            device=device,
        )
        wp.launch(
            _pair_counts_from_starts_kernel,
            dim=capacity,
            inputs=[count_arr, num_pairs, pair_first],
            outputs=[pair_count],
            device=device,
        )
        self.assertEqual(int(num_pairs.numpy()[0]), 2)
        np.testing.assert_array_equal(pair_shape_a.numpy()[:2], [1, 4])
        np.testing.assert_array_equal(pair_shape_b.numpy()[:2], [2, 5])
        np.testing.assert_array_equal(pair_count.numpy()[:2], [3, 2])
        np.testing.assert_array_equal(pair_first.numpy()[:2], [0, 3])


if __name__ == "__main__":
    wp.init()
    unittest.main()
