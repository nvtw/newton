# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the mass-splitting integration in
:class:`PhoenXWorld`.

Sub-step 6b verifies the build path: with ``mass_splitting=True``, the
per-step substep loop builds the (body, partition_key) interaction
graph after coloring and writes it into the copy-state container.
The broadcast / average / writeback wiring lands in 6c.

For the build to produce entries, the world needs at least one active
constraint that involves a non-static body. We construct a hand-built
two-body scene with a single joint that exercises the partitioner +
emit path.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _make_minimal_world(*, mass_splitting: bool, device) -> PhoenXWorld:
    """2-body, no joint, no contact world. step() runs but the
    interaction graph stays empty — exercise the no-op path."""
    bodies = body_container_zeros(2, device=device)
    constraints = PhoenXWorld.make_constraint_container(num_joints=0, device=device)
    return PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        num_joints=0,
        rigid_contact_max=4,
        mass_splitting=mass_splitting,
        max_colored_partitions=12,
        step_layout="single_world",
        device=device,
    )


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX mass-splitting tests are CUDA-only per feedback_phoenx_tests_capture_only.",
)
class TestMassSplittingSolverWiring(unittest.TestCase):
    def test_step_runs_with_mass_splitting_enabled(self):
        # Bare-bones smoke test: step() must not crash when
        # ``mass_splitting=True`` even with zero active constraints.
        # The build kernel still launches (single-world path) but
        # ``num_active_constraints == 0`` makes the emit a no-op.
        device = wp.get_preferred_device()
        world = _make_minimal_world(mass_splitting=True, device=device)
        world.step(0.01)
        wp.synchronize_device(device)
        # No constraints → no pairs emitted → graph stays disabled
        # (``highest_index_in_use[0] == 0``).
        self.assertEqual(int(world._copy_state.highest_index_in_use.numpy()[0]), 0)

    def test_step_runs_under_graph_capture(self):
        # Load-bearing capture-safety assertion: every kernel launched
        # by the new ``_rebuild_mass_splitting_graph`` path must work
        # inside ``wp.ScopedCapture`` + ``wp.capture_launch``. See
        # ``feedback_phoenx_tests_capture_only.md``.
        device = wp.get_preferred_device()
        world = _make_minimal_world(mass_splitting=True, device=device)
        # Warm-up.
        world.step(0.01)
        with wp.ScopedCapture(device=device) as capture:
            world.step(0.01)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

    def test_build_populates_copy_state_for_hand_built_elements(self):
        # Bypass contact ingest: stuff three "fake" constraint elements
        # directly into the partitioner's element buffer, color them,
        # and run the mass-splitting build. Assert each non-static body
        # endpoint shows up exactly once in the copy state with
        # ``inv_factor == 1`` (one slot per body since each lives in a
        # single colour bucket).
        device = wp.get_preferred_device()
        bodies = body_container_zeros(4, device=device)
        # Mark body 0 static (inverse_mass=0); bodies 1, 2, 3 are dynamic.
        bodies.inverse_mass.assign(np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32))
        constraints = PhoenXWorld.make_constraint_container(num_joints=0, device=device)
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            rigid_contact_max=4,
            mass_splitting=True,
            max_colored_partitions=12,
            step_layout="single_world",
            device=device,
        )

        # Inject three hand-built elements directly into the partitioner
        # input. Each element has 2 non-static body endpoints; element
        # 0 conflicts with element 1 via body 1, so the coloring will
        # put them in different buckets.
        max_bodies = int(MAX_BODIES)
        struct_dtype = np.dtype(
            {"names": ["bodies"], "formats": [(np.int32, max_bodies)], "offsets": [0], "itemsize": 32}
        )
        elems_np = np.zeros(world._constraint_capacity, dtype=struct_dtype)
        elems_np["bodies"][:] = -1
        elems_np["bodies"][0, 0] = 1
        elems_np["bodies"][0, 1] = 2
        elems_np["bodies"][1, 0] = 1
        elems_np["bodies"][1, 1] = 3
        elems_np["bodies"][2, 0] = 2
        elems_np["bodies"][2, 1] = 3
        world._elements.assign(wp.from_numpy(elems_np, dtype=ElementInteractionData, device=device))
        world._num_active_constraints.assign(np.array([3], dtype=np.int32))

        # Run the partitioner + mass-splitting build by hand (skip
        # step()'s contact ingest path).
        world._partitioner.reset(world._elements, world._num_active_constraints)
        world._partitioner.build_csr_greedy_with_jp_fallback()
        world._rebuild_mass_splitting_graph()
        wp.synchronize_device(device)

        # With max_colored_partitions=12 and only 3 elements, coloring
        # fits trivially in 3 colours -- no overflow. Every body lives
        # in exactly one colour bucket -> one (body, partition_key=0)
        # slot per body.
        section_end = world._copy_state.section_end.numpy()
        # Body 0 is static -> 0 slots. Bodies 1, 2, 3 -> 1 slot each.
        self.assertEqual(section_end[0], 0)
        self.assertEqual(section_end[1] - section_end[0], 1)
        self.assertEqual(section_end[2] - section_end[1], 1)
        self.assertEqual(section_end[3] - section_end[2], 1)
        # 3 slots total (1 per dynamic body).
        self.assertEqual(int(world._copy_state.highest_index_in_use.numpy()[0]), 3)
        # Every slot's partition_key is 0 (no overflow).
        partition_list = world._copy_state.partition_list.numpy()[:3]
        np.testing.assert_array_equal(partition_list, [0, 0, 0])

    def test_overflow_bucket_produces_multiple_slots_per_body(self):
        # Force coloring into the overflow bucket by setting
        # ``max_colored_partitions=1``: every adjacent element after
        # the first one lands in colour 1 (overflow). A hub body
        # touching three elements ends up with multiple slots.
        device = wp.get_preferred_device()
        bodies = body_container_zeros(5, device=device)
        bodies.inverse_mass.assign(np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        constraints = PhoenXWorld.make_constraint_container(num_joints=0, device=device)
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            rigid_contact_max=8,
            mass_splitting=True,
            max_colored_partitions=1,  # 1 regular colour + 1 overflow
            step_layout="single_world",
            device=device,
        )

        max_bodies = int(MAX_BODIES)
        struct_dtype = np.dtype(
            {"names": ["bodies"], "formats": [(np.int32, max_bodies)], "offsets": [0], "itemsize": 32}
        )
        elems_np = np.zeros(world._constraint_capacity, dtype=struct_dtype)
        elems_np["bodies"][:] = -1
        # Body 1 is the hub; three elements share it.
        elems_np["bodies"][0, 0] = 1
        elems_np["bodies"][0, 1] = 2
        elems_np["bodies"][1, 0] = 1
        elems_np["bodies"][1, 1] = 3
        elems_np["bodies"][2, 0] = 1
        elems_np["bodies"][2, 1] = 4
        world._elements.assign(wp.from_numpy(elems_np, dtype=ElementInteractionData, device=device))
        world._num_active_constraints.assign(np.array([3], dtype=np.int32))

        world._partitioner.reset(world._elements, world._num_active_constraints)
        world._partitioner.build_csr_greedy_with_jp_fallback()
        world._rebuild_mass_splitting_graph()
        wp.synchronize_device(device)

        # One element colours at 0 (regular), the other two land in
        # colour 1 (overflow). Body 1 (the hub) participates in all
        # three, so it gets >= 2 slots (1 for the regular bucket, then
        # one per overflow slot it touches with batch_size=1).
        section_end = world._copy_state.section_end.numpy()
        body1_slot_count = section_end[1] - section_end[0]
        self.assertGreaterEqual(body1_slot_count, 2)
        # Each non-hub body (2, 3, 4) participates in exactly one
        # element, so they get exactly 1 slot each.
        self.assertEqual(section_end[2] - section_end[1], 1)
        self.assertEqual(section_end[3] - section_end[2], 1)
        self.assertEqual(section_end[4] - section_end[3], 1)
        # Static body has 0 slots.
        self.assertEqual(section_end[0], 0)

    def test_disabled_path_unchanged(self):
        # ``mass_splitting=False`` (default) must not even launch the
        # build kernel; ``_copy_state.highest_index_in_use`` stays at
        # the sentinel zero and ``_interaction_graph_scratch.num_pairs``
        # is never written to.
        device = wp.get_preferred_device()
        world = _make_minimal_world(mass_splitting=False, device=device)
        world.step(0.01)
        wp.synchronize_device(device)
        self.assertEqual(int(world._copy_state.highest_index_in_use.numpy()[0]), 0)
        self.assertEqual(int(world._interaction_graph_scratch.num_pairs.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main()
