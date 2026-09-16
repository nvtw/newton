# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify explicit color-group partitions preserve natural endpoint slots."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints import constraint_contact as cc
from newton._src.solvers.phoenx.constraints import constraint_container as cs
from newton._src.solvers.phoenx.mass_splitting.copy_state import copy_state_container_zeros
from newton._src.solvers.phoenx.mass_splitting.slot_cache import (
    build_partition_slot_cache_kernel,
    build_slot_cache_kernel,
)


@wp.kernel
def contact_fixture(c: cc.ContactColumnContainer):
    cc.contact_set_body1(c, 0, 1)
    cc.contact_set_body2(c, 0, 4)
    cc.contact_set_side0_nodes_extra(c, 0, wp.vec3i(-1, 2, 99))
    cc.contact_set_side1_nodes_extra(c, 0, wp.vec3i(0, 3, -1))


@wp.kernel
def read_contact_cache(c: cc.ContactColumnContainer, slots: wp.array[wp.int32], counts: wp.array[wp.int32]):
    slots[0] = cc.contact_get_slot1(c, 0)
    slots[1] = cc.contact_get_slot2(c, 0)
    counts[0] = cc.contact_get_count1(c, 0)
    counts[1] = cc.contact_get_count2(c, 0)
    for i in range(3):
        slots[2 + i] = cc.contact_get_side0_slots_extra(c, 0)[i]
        slots[5 + i] = cc.contact_get_side1_slots_extra(c, 0)[i]
        counts[2 + i] = cc.contact_get_side0_counts_extra(c, 0)[i]
        counts[5 + i] = cc.contact_get_side1_counts_extra(c, 0)[i]


class TestPartitionSlotCache(unittest.TestCase):
    def _check_natural_slots(self, device):
        """Retain pinned endpoint ordinals, missing-node fallbacks and legacy partitions."""
        constraints = cs.constraint_container_zeros(5, 9, device)
        types = [
            cs.CONSTRAINT_TYPE_CLOTH_TRIANGLE,
            cs.CONSTRAINT_TYPE_CLOTH_BENDING,
            cs.CONSTRAINT_TYPE_SOFT_TETRAHEDRON,
            cs.CONSTRAINT_TYPE_SOFT_TETRAHEDRON_NEOHOOKEAN,
            cs.CONSTRAINT_TYPE_SOFT_HEXAHEDRON,
        ]
        bodies = [0, -1, 2, 3, 4, -1, 99, 1]
        host = constraints.data.numpy()
        host.view(np.int32)[0] = types
        host.view(np.int32)[1:9] = np.tile(np.asarray(bodies)[:, None], (1, 5))
        constraints.data.assign(host)
        contacts = cc.contact_column_container_zeros(1, device)
        wp.launch(contact_fixture, 1, [contacts], device=device)
        keys = [[0, 2], [], [1, 2, 3], [0, 1, 4], [1, 4]]
        flat = [key for group in keys for key in group]
        copies = copy_state_container_zeros(len(flat), len(keys), device)
        sizes = np.asarray([len(group) for group in keys], dtype=np.int32)
        ends = np.cumsum(sizes, dtype=np.int32)
        copies.section_end.assign(ends)
        copies.count_per_node.assign(sizes)
        copies.partition_list.assign(np.asarray(flat, dtype=np.int32))
        copies.slot_for_pid0.assign(
            np.asarray([int(ends[i] - sizes[i]) if 0 in group else -1 for i, group in enumerate(keys)], dtype=np.int32)
        )
        copies.highest_index_in_use.assign(np.asarray([len(flat)], dtype=np.int32))
        order = np.asarray([5, 2, 0, 4, 3, 1], dtype=np.int32)
        ids = wp.array(order, dtype=wp.int32, device=device)
        active = wp.array([6], dtype=wp.int32, device=device)
        slots = wp.zeros(8, dtype=wp.int32, device=device)
        counts = wp.zeros(8, dtype=wp.int32, device=device)
        for legacy in (False, True):
            partitions = np.asarray([0, 1, 2, 3, 4, 1], dtype=np.int32)
            if legacy:
                partitions[order] = [0, 0, 0, 0, 1, 1]
            row_partition = wp.array(partitions, dtype=wp.int32, device=device)
            wp.launch(
                build_partition_slot_cache_kernel,
                8,
                [ids, row_partition, active, copies, constraints, contacts, 5],
                device=device,
            )

            def expected(nodes, partition):
                result = []
                for node in nodes:
                    if node < 0 or node >= len(keys) or partition not in keys[node]:
                        result.append((-1, 1))
                    else:
                        result.append((int(ends[node] - sizes[node]) + keys[node].index(partition), len(keys[node])))
                return np.asarray(result).T

            for cid, n in enumerate((3, 4, 4, 4, 8)):
                expected_slots, expected_counts = expected(bodies[:n], int(partitions[cid]))
                np.testing.assert_array_equal(constraints.slot_cache.numpy()[cid, :n], expected_slots)
                np.testing.assert_array_equal(constraints.count_cache.numpy()[cid, :n], expected_counts)
            wp.launch(read_contact_cache, 1, [contacts, slots, counts], device=device)
            expected_slots, expected_counts = expected([1, 4, -1, 2, 99, 0, 3, -1], int(partitions[5]))
            np.testing.assert_array_equal(slots.numpy(), expected_slots)
            np.testing.assert_array_equal(counts.numpy(), expected_counts)
            if legacy:
                before = [constraints.slot_cache.numpy(), constraints.count_cache.numpy(), contacts.data.numpy()]
                starts = wp.array([0, 2, 6], dtype=wp.int32, device=device)
                wp.launch(
                    build_slot_cache_kernel,
                    8,
                    [ids, starts, active, copies, constraints, contacts, 5, 1, 2],
                    device=device,
                )
                for actual, wanted in zip(
                    (constraints.slot_cache.numpy(), constraints.count_cache.numpy(), contacts.data.numpy()),
                    before,
                    strict=True,
                ):
                    np.testing.assert_array_equal(actual, wanted)

    def test_cpu_natural_slots_and_legacy_equivalence(self):
        """Verify CPU endpoint mapping against independent expected slots."""
        self._check_natural_slots("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is unavailable")
    def test_cuda_natural_slots_and_legacy_equivalence(self):
        """Verify GPU endpoint mapping against independent expected slots."""
        self._check_natural_slots("cuda:0")


if __name__ == "__main__":
    unittest.main()
