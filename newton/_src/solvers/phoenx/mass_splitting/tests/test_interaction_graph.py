# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the mass-splitting interaction graph builder.

Tests run inside ``wp.ScopedCapture`` per
``feedback_phoenx_tests_capture_only.md``. The build pipeline must be
CUDA-graph compatible (radix_sort_pairs + array_scan + custom launches),
so capture-launch is the load-bearing assertion.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.mass_splitting.copy_state import (
    copy_state_container_zeros,
)
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    InteractionGraphScratch,
    build_interaction_graph,
    emit_pair,
    interaction_graph_scratch_zeros,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _seed_packed_keys_kernel(
    packed_keys: wp.array[wp.int64],
    node_ids: wp.array[wp.int32],
    partition_keys: wp.array[wp.int32],
    num_pairs: wp.array[wp.int32],
):
    """Directly write ``packed_keys[i] = pack(node_ids[i], partition_keys[i])``.

    Used by tests that want deterministic pre-sort ordering. The
    production emit path goes through :func:`emit_pair` (atomic-add),
    which loses ordering but is correctness-equivalent post-sort.
    """
    tid = wp.tid()
    if tid >= node_ids.shape[0]:
        return
    n = node_ids[tid]
    p = partition_keys[tid]
    key = (wp.int64(n) << wp.int64(32)) | (wp.int64(p) & wp.int64(0xFFFFFFFF))
    packed_keys[tid] = key
    if tid == 0:
        num_pairs[0] = node_ids.shape[0]


@wp.kernel(enable_backward=False)
def _emit_via_atomic_kernel(
    scratch: InteractionGraphScratch,
    node_ids: wp.array[wp.int32],
    partition_keys: wp.array[wp.int32],
):
    """Exercise the production :func:`emit_pair` atomic-counter path."""
    tid = wp.tid()
    if tid >= node_ids.shape[0]:
        return
    emit_pair(scratch, node_ids[tid], partition_keys[tid])


def _seed_pairs_direct(
    scratch: InteractionGraphScratch,
    pairs: list[tuple[int, int]],
    device,
) -> None:
    """Direct-write path: skip the atomic emit, stamp packed_keys at
    deterministic slots, write num_pairs.
    """
    n = len(pairs)
    node_ids = wp.from_numpy(np.asarray([p[0] for p in pairs], dtype=np.int32), dtype=wp.int32, device=device)
    partition_keys = wp.from_numpy(np.asarray([p[1] for p in pairs], dtype=np.int32), dtype=wp.int32, device=device)
    wp.launch(
        _seed_packed_keys_kernel,
        dim=n,
        inputs=[scratch.packed_keys, node_ids, partition_keys, scratch.num_pairs],
        device=device,
    )


def _build_and_read(
    pairs: list[tuple[int, int]],
    capacity: int,
    num_nodes: int,
    device,
    use_atomic_emit: bool = False,
) -> dict:
    """Allocate scratch + copy_state, seed pairs, run build under a captured
    graph, return numpy snapshots of the build outputs.
    """
    cs = copy_state_container_zeros(capacity=capacity, num_nodes=num_nodes, device=device)
    scratch = interaction_graph_scratch_zeros(capacity=capacity, device=device)

    if use_atomic_emit:
        n = len(pairs)
        node_ids = wp.from_numpy(np.asarray([p[0] for p in pairs], dtype=np.int32), dtype=wp.int32, device=device)
        partition_keys = wp.from_numpy(np.asarray([p[1] for p in pairs], dtype=np.int32), dtype=wp.int32, device=device)

        # Warm-up to JIT-compile kernels outside the capture.
        wp.launch(
            _emit_via_atomic_kernel,
            dim=n,
            inputs=[scratch, node_ids, partition_keys],
            device=device,
        )
        scratch.num_pairs.zero_()
        # zero_() to reset emit counter pre-capture; build's own
        # _reset_num_pairs_kernel runs *after* the build, so we need
        # to reset before capture so the emit lands the right values.

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                _emit_via_atomic_kernel,
                dim=n,
                inputs=[scratch, node_ids, partition_keys],
                device=device,
            )
            build_interaction_graph(scratch, cs)
    else:
        _seed_pairs_direct(scratch, pairs, device)
        # Build inside ScopedCapture.
        with wp.ScopedCapture(device=device) as capture:
            build_interaction_graph(scratch, cs)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    return {
        "section_end": cs.section_end.numpy().copy(),
        "partition_list": cs.partition_list.numpy().copy(),
        "highest_index_in_use": int(cs.highest_index_in_use.numpy()[0]),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX mass-splitting tests are CUDA-only per feedback_phoenx_tests_capture_only.",
)
class TestInteractionGraphBuild(unittest.TestCase):
    def test_single_body_single_partition(self):
        # One body (id 0), one partition (key 7). Should yield one slot
        # in section [0, 1], partition_list[0] == 7.
        device = wp.get_preferred_device()
        result = _build_and_read(pairs=[(0, 7)], capacity=8, num_nodes=1, device=device)
        self.assertEqual(result["highest_index_in_use"], 1)
        np.testing.assert_array_equal(result["section_end"], [1])
        self.assertEqual(int(result["partition_list"][0]), 7)

    def test_dedup_within_node(self):
        # Three duplicates of (node=2, partition=3) should collapse to one slot.
        device = wp.get_preferred_device()
        result = _build_and_read(pairs=[(2, 3), (2, 3), (2, 3)], capacity=8, num_nodes=4, device=device)
        self.assertEqual(result["highest_index_in_use"], 1)
        # Nodes 0,1 are empty (count 0 → section_end stays 0 after inclusive sum scan).
        # Node 2 has 1 entry; sum scan to that point = 1. Node 3 empty so it
        # carries 1 forward.
        np.testing.assert_array_equal(result["section_end"], [0, 0, 1, 1])
        self.assertEqual(int(result["partition_list"][0]), 3)

    def test_two_bodies_three_partitions_each(self):
        # Body 0 has partition keys {0, 5, 7}. Body 1 has partition keys {0, 2}.
        # After sort: (0,0), (0,5), (0,7), (1,0), (1,2). section_end = [3, 5].
        device = wp.get_preferred_device()
        pairs = [(0, 0), (0, 5), (0, 7), (1, 0), (1, 2)]
        result = _build_and_read(pairs=pairs, capacity=16, num_nodes=2, device=device)
        self.assertEqual(result["highest_index_in_use"], 5)
        np.testing.assert_array_equal(result["section_end"], [3, 5])
        # partition_list is grouped by node, sorted ascending within each node
        # by the packed key's low 32 bits.
        np.testing.assert_array_equal(result["partition_list"][:5], [0, 5, 7, 0, 2])

    def test_empty_nodes_inherit_predecessor(self):
        # Body 0 has 1 entry, body 1 empty, body 2 has 1 entry.
        # Counts: [1, 0, 1] → inclusive scan: [1, 1, 2].
        device = wp.get_preferred_device()
        result = _build_and_read(pairs=[(0, 4), (2, 9)], capacity=8, num_nodes=3, device=device)
        self.assertEqual(result["highest_index_in_use"], 2)
        np.testing.assert_array_equal(result["section_end"], [1, 1, 2])
        np.testing.assert_array_equal(result["partition_list"][:2], [4, 9])

    def test_zero_pairs_disabled_path(self):
        # No pairs emitted → highest_index_in_use stays 0 (the disabled
        # fast-path probe) and section_end is all zeros.
        device = wp.get_preferred_device()
        cs = copy_state_container_zeros(capacity=8, num_nodes=4, device=device)
        scratch = interaction_graph_scratch_zeros(capacity=8, device=device)

        # Warm-up.
        build_interaction_graph(scratch, cs)
        with wp.ScopedCapture(device=device) as capture:
            build_interaction_graph(scratch, cs)
        wp.capture_launch(capture.graph)

        self.assertEqual(int(cs.highest_index_in_use.numpy()[0]), 0)
        np.testing.assert_array_equal(cs.section_end.numpy(), 0)

    def test_atomic_emit_produces_same_result_as_direct_seed(self):
        # Same pair set, two emit paths: direct write vs production
        # atomic emit_pair. Post-sort outputs must match.
        device = wp.get_preferred_device()
        pairs = [(0, 0), (0, 5), (1, 2), (2, 1), (2, 9), (3, 0)]
        a = _build_and_read(pairs=pairs, capacity=16, num_nodes=4, device=device, use_atomic_emit=False)
        b = _build_and_read(pairs=pairs, capacity=16, num_nodes=4, device=device, use_atomic_emit=True)
        self.assertEqual(a["highest_index_in_use"], b["highest_index_in_use"])
        np.testing.assert_array_equal(a["section_end"], b["section_end"])
        np.testing.assert_array_equal(a["partition_list"], b["partition_list"])

    def test_relaunch_clears_prior_build(self):
        # Capture once, launch twice with different pair sets seeded
        # before each launch. The second launch must overwrite the
        # first's outputs (the build resets section_end + dest_idx +
        # is_boundary at the top, so re-running is safe).
        device = wp.get_preferred_device()
        cs = copy_state_container_zeros(capacity=16, num_nodes=4, device=device)
        scratch = interaction_graph_scratch_zeros(capacity=16, device=device)

        # First seed.
        _seed_pairs_direct(scratch, [(0, 1), (1, 2), (2, 3), (3, 4)], device)
        # Warm-up.
        build_interaction_graph(scratch, cs)

        with wp.ScopedCapture(device=device) as capture:
            build_interaction_graph(scratch, cs)

        # Launch with original seed.
        _seed_pairs_direct(scratch, [(0, 1), (1, 2), (2, 3), (3, 4)], device)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(cs.section_end.numpy(), [1, 2, 3, 4])
        self.assertEqual(int(cs.highest_index_in_use.numpy()[0]), 4)

        # Re-seed differently and relaunch the SAME captured graph.
        _seed_pairs_direct(scratch, [(0, 7), (0, 8), (3, 0)], device)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(cs.section_end.numpy(), [2, 2, 2, 3])
        self.assertEqual(int(cs.highest_index_in_use.numpy()[0]), 3)


if __name__ == "__main__":
    unittest.main()
