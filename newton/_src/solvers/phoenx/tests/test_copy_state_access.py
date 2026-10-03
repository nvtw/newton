# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check copy-state slot lookup for direct, regular and overflow partitions."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.mass_splitting.access import get_state_index
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer, copy_state_container_zeros
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    build_interaction_graph,
    interaction_graph_scratch_zeros,
)


@wp.kernel
def _lookup_kernel(
    copy_state: CopyStateContainer,
    nodes: wp.array[wp.int32],
    partitions: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    factors: wp.array[wp.int32],
):
    i = wp.tid()
    slot, factor = get_state_index(copy_state, nodes[i], partitions[i])
    slots[i] = slot
    factors[i] = factor


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX mass splitting runs on CUDA")
class TestCopyStateAccess(unittest.TestCase):
    def test_direct_partitions_bypass_owned_slots(self):
        """Direct colours read body storage even when overflow copies exist."""
        device = "cuda:0"
        pairs = [(0, 0), (0, 3), (1, 3), (2, 1), (2, 2), (2, 5)]
        scratch = interaction_graph_scratch_zeros(16, device=device)
        copy_state = copy_state_container_zeros(16, 4, device=device)
        keys = np.zeros(32, dtype=np.int64)
        keys[: len(pairs)] = [(node << 32) | partition for node, partition in pairs]
        scratch.packed_keys.assign(keys)
        scratch.num_pairs.assign(np.array([len(pairs)], dtype=np.int32))
        build_interaction_graph(scratch, copy_state)

        queries = [(0, -1), (2, -1), (3, -1), (0, 3), (2, 2), (2, 4), (1, 3), (3, 0)]
        nodes = wp.array([node for node, _ in queries], dtype=wp.int32, device=device)
        partitions = wp.array([partition for _, partition in queries], dtype=wp.int32, device=device)
        slots = wp.zeros(len(queries), dtype=wp.int32, device=device)
        factors = wp.zeros(len(queries), dtype=wp.int32, device=device)
        wp.launch(_lookup_kernel, len(queries), [copy_state, nodes, partitions, slots, factors], device=device)

        section_end = copy_state.section_end.numpy()
        partition_list = copy_state.partition_list.numpy()
        for (node, partition), slot, factor in zip(queries, slots.numpy(), factors.numpy(), strict=True):
            start = 0 if node == 0 else int(section_end[node - 1])
            owned = partition_list[start : int(section_end[node])].tolist()
            if partition >= 0 and partition in owned:
                self.assertEqual(int(partition_list[slot]), partition)
                self.assertEqual(int(factor), len(owned))
            else:
                self.assertEqual((int(slot), int(factor)), (-1, 1))


if __name__ == "__main__":
    unittest.main()
