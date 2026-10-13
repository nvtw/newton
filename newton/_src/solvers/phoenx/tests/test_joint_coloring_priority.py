# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep structural joints out of overflow when contacts compete for colors."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_column_container_zeros,
    contact_set_contact_count,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    _pack_constraint_priority,
    _priority_is_greater,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import IncrementalContactPartitioner
from newton._src.solvers.phoenx.graph_coloring.luby_fixed import FixedIterationLubyPartitioner
from newton._src.solvers.phoenx.simulation_kernels import _cost_biased_priority
from newton._src.solvers.phoenx.tests.test_graph_coloring_speculative import _make_elements
from newton._src.solvers.phoenx.tests.test_joint_row_capacity import _make_model


@wp.kernel
def _set_counts(columns: ContactColumnContainer, counts: wp.array[wp.int32]):
    i = wp.tid()
    contact_set_contact_count(columns, i, counts[i])


@wp.kernel
def _compare_priorities(priorities: wp.array[wp.int32], result: wp.array[wp.int32]):
    i = wp.tid()
    if _priority_is_greater(priorities[0], wp.int32(0), priorities[i + 1], i + 1):
        result[i] = 1
    if _cost_biased_priority(priorities, wp.int32(0)) > _cost_biased_priority(priorities, i + 1):
        result[i + 3] = 1


@wp.kernel
def _pack_constraints(costs: wp.array[wp.int32], random_values: wp.array[wp.int32], result: wp.array[wp.int32]):
    i = wp.tid()
    result[i] = _pack_constraint_priority(costs[i], random_values[i], i == wp.int32(0))


class TestJointColoringPriority(unittest.TestCase):
    def test_solver_preserves_default_joint_order(self):
        """Preserve convergent default ordering and make joint priority explicit."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        for options, expected in (({}, 0), ({"joint_coloring_priority": True}, 255)):
            with self.subTest(options=options):
                model = _make_model(["hinge", "drive"])
                solver = newton.solvers.SolverPhoenX(
                    model,
                    joint_mode="maximal_pgs",
                    step_layout="single_world",
                    substeps=1,
                    solver_iterations=1,
                    velocity_iterations=1,
                    mass_splitting=True,
                    mass_splitting_overflow_only=True,
                    mass_splitting_unrolled=True,
                    mass_splitting_batch_size=1,
                    max_colored_partitions=9,
                    partitioner_algorithm="endpoint_owner",
                    **options,
                )
                state = model.state()
                solver.step(state, state, model.control(), None, 1.0e-6)
                cids = solver._joint_constraints.joint_idx_to_cid.numpy()
                cids = cids[cids >= 0]
                priorities = solver.world._partitioner._packed_priorities.numpy().view(np.uint32)
                self.assertEqual(len(cids), 2)
                np.testing.assert_array_equal(priorities[cids] >> 24, np.full(len(cids), expected))

    def test_constraint_priority_classes(self):
        """Reserve the top cost for joints without changing random bits."""
        costs = wp.array([0, 1, 128, 255, 300, 0], dtype=wp.int32, device="cpu")
        random_values = wp.array([1, 2, 3, 4, 5, 0x01FFFFFF], dtype=wp.int32, device="cpu")
        result = wp.zeros(6, dtype=wp.int32, device="cpu")
        wp.launch(_pack_constraints, dim=6, inputs=[costs, random_values, result], device="cpu")
        packed = result.numpy().view(np.uint32)
        np.testing.assert_array_equal(packed >> 24, [255, 1, 128, 254, 254, 0])
        np.testing.assert_array_equal(packed & 0x00FFFFFF, [1, 2, 3, 4, 5, 0x00FFFFFF])

    def test_unsigned_priority_order(self):
        """The high cost byte must retain its order across the sign bit."""
        packed = np.array([0xFF000001, 0x01000002, 0x80000003, 0xFE000004], dtype=np.uint32).view(np.int32)
        result = wp.zeros(6, dtype=wp.int32, device="cpu")
        wp.launch(_compare_priorities, dim=3, inputs=[wp.array(packed, device="cpu"), result], device="cpu")
        np.testing.assert_array_equal(result.numpy(), np.ones(6, dtype=np.int32))

    def test_joint_priority_survives_dense_contacts(self):
        """Five star joints take the five available colors before contacts."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        device = "cuda:0"
        joints = 5
        contacts = 20
        size = joints + contacts
        edges = np.full((size, 8), -1, dtype=np.int32)
        edges[:joints, 0] = 0
        edges[:joints, 1] = np.arange(1, joints + 1)
        edges[joints:, 0] = 0
        edges[joints:, 1] = np.arange(joints + 1, size + 1)
        elements = _make_elements(edges, device)
        count = wp.array([size], dtype=wp.int32, device=device)
        columns = contact_column_container_zeros(contacts, device=device)
        contact_counts = wp.array([1, 128, 254, 255, 300] * 4, dtype=wp.int32, device=device)
        wp.launch(_set_counts, dim=contacts, inputs=[columns, contact_counts], device=device)
        for algorithm in ("jp", "greedy", "speculative", "endpoint_owner", "luby_fixed"):
            with self.subTest(algorithm=algorithm):
                if algorithm == "luby_fixed":
                    partitioner = FixedIterationLubyPartitioner(
                        size, size + 1, device=device, max_colored_partitions=joints
                    )
                else:
                    partitioner = IncrementalContactPartitioner(
                        size,
                        size + 1,
                        device=device,
                        enable_warm_start=False,
                        max_colored_partitions=joints,
                        endpoint_owner_coloring=algorithm == "endpoint_owner",
                    )
                partitioner.set_costs_from_contacts(
                    joints, wp.array([contacts], dtype=wp.int32, device=device), columns
                )
                if algorithm != "luby_fixed":
                    partitioner.set_speculative_coloring(algorithm == "speculative")
                previous = None
                for _ in range(3):
                    partitioner.reset(elements, count)
                    if algorithm in ("jp", "luby_fixed"):
                        partitioner.build_csr()
                    elif algorithm == "endpoint_owner":
                        partitioner.build_csr_endpoint_owner()
                    else:
                        partitioner.build_csr_greedy_with_jp_fallback()
                    assignments = partitioner.interaction_id_to_partition.numpy()[:size]
                    np.testing.assert_array_equal(np.sort(assignments[:joints]), np.arange(joints))
                    self.assertTrue(np.all(assignments[joints:] == joints))
                    starts = partitioner.color_starts.numpy()
                    ids = partitioner.element_ids_by_color.numpy()[:size]
                    np.testing.assert_array_equal(np.sort(ids), np.arange(size))
                    self.assertEqual(int(starts[joints + 1]), size)
                    if previous is not None:
                        np.testing.assert_array_equal(assignments, previous)
                    previous = assignments.copy()


if __name__ == "__main__":
    unittest.main()
