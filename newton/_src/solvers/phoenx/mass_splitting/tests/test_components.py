# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-component unit tests for the mass-splitting pipeline.

Each test validates ONE kernel / function in isolation against a
small synthetic input with a hand-computable expected output.
Cross-references the C# reference implementation in
``experimentalsim/PhoenX/src/PhoenX/CudaKernels/`` so any drift
shows up here before it shows up in the integration tests.

Tests are CUDA-only and run inside ``wp.ScopedCapture`` per the
PhoenX testing convention.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    ElementInteractionData,
    vec8i,
)
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    InteractionGraph,
    InteractionGraphData,
    graph_get_rigid_state_index,
)
from newton._src.solvers.phoenx.mass_splitting.read_state import read_state, write_state
from newton._src.solvers.phoenx.mass_splitting.setup_kernels import (
    average_and_broadcast_unified_kernel,
    broadcast_to_copy_states_unified_kernel,
    build_partition_list_kernel,
    compact_unique_keys_kernel,
    copy_state_into_unified_kernel,
    mark_unique_keys_kernel,
    prefix_max_scan_kernel,
    record_all_interactions_kernel,
    reset_construction_buffers_kernel,
    write_highest_index_kernel,
)
from newton._src.solvers.phoenx.mass_splitting.state import (
    ACCESS_MODE_VELOCITY_LEVEL,
    TinyRigidState,
)
from newton._src.solvers.phoenx.particle import ParticleContainer, particle_container_zeros


CUDA_ONLY = unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX kernels run on CUDA only.")


# ---------------------------------------------------------------------------
# Helpers: build synthetic InteractionGraphData / construction buffers /
# elements arrays on device.
# ---------------------------------------------------------------------------


def _make_graph_with_entries(entries, *, max_rigid_bodies, max_interactions, device):
    """Build an InteractionGraph populated with ``(body, partition)``
    entries on the host, then ``build()`` to upload."""
    graph = InteractionGraph(
        max_rigid_bodies=max_rigid_bodies,
        max_interactions=max_interactions,
        device=device,
    )
    for body, partition in entries:
        graph.add_entry(body, partition)
    graph.build()
    return graph


def _make_elements_array(rows, *, capacity, device):
    """``rows`` is a list of body-id tuples (length up to 8). Pads
    with -1 to 8 slots and returns the device array."""
    np_rows = np.full((capacity, 8), -1, dtype=np.int32)
    for i, row in enumerate(rows):
        for j, b in enumerate(row):
            np_rows[i, j] = int(b)
    arr = wp.zeros(capacity, dtype=ElementInteractionData, device=device)
    # ElementInteractionData is a wp.struct with `bodies: vec8i`.
    # Populate via a small kernel.

    @wp.kernel(enable_backward=False)
    def _populate(out: wp.array[ElementInteractionData], data: wp.array2d[wp.int32]):
        i = wp.tid()
        e = ElementInteractionData()
        e.bodies = vec8i(
            data[i, 0], data[i, 1], data[i, 2], data[i, 3],
            data[i, 4], data[i, 5], data[i, 6], data[i, 7],
        )
        out[i] = e

    data_dev = wp.array2d(np_rows, dtype=wp.int32, device=device)
    wp.launch(_populate, dim=capacity, inputs=[arr, data_dev], device=device)
    return arr


# ---------------------------------------------------------------------------
# Test 1: cid_to_partition_constraint_id matches C# ParallelId.
# ---------------------------------------------------------------------------


@CUDA_ONLY
class TestCidToPartitionConstraintId(unittest.TestCase):
    """C# ``MassSplittingTypes.cuh:80-87`` ``ParallelId(isAdditional, offset)``:

    * regular partition (isAdditional=False) -> ``return 0``
    * overflow (isAdditional=True) -> ``(globalConstraintId + offset) / batchSize``

    Newton's ``record_all_interactions_kernel`` writes the same
    value to ``cid_to_partition_constraint_id[cid]``: 0 for cids in
    colours ``< max_partitions`` (regular), and ``j / batch_size``
    for cids in colour ``max_partitions`` (overflow), where ``j`` is
    the cid's offset into the overflow bucket of the CSR.
    """

    def test_pcid_zero_for_regular_overflow_position_for_overflow(self):
        device = wp.get_preferred_device()
        max_partitions = 3
        batch_size = 1
        # 4 colours total: 0, 1, 2 (regular) + 3 (overflow). Layout:
        #   color 0: cids [0]   (1 entry)
        #   color 1: cids [1]   (1 entry)
        #   color 2: cids [2]   (1 entry)
        #   color 3: cids [3, 4, 5]  (overflow, 3 entries)
        n_active = 6
        elements = _make_elements_array(
            [(0,), (1,), (2,), (3,), (4,), (5,)],
            capacity=n_active, device=device,
        )
        interaction_id_to_partition = wp.array(
            np.array([0, 1, 2, 3, 3, 3], dtype=np.int32),
            dtype=wp.int32, device=device,
        )
        element_ids_by_color = wp.array(
            np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
            dtype=wp.int32, device=device,
        )
        color_starts = wp.array(
            np.array([0, 1, 2, 3, 6], dtype=np.int32),
            dtype=wp.int32, device=device,
        )
        num_active_constraints = wp.array([n_active], dtype=wp.int32, device=device)
        # Construction buffers (we don't validate them in this test;
        # just need the kernel call to be side-effect safe).
        construction_keys = wp.zeros(64, dtype=wp.int64, device=device)
        construction_count = wp.zeros(1, dtype=wp.int32, device=device)
        cid_to_partition_constraint_id = wp.zeros(n_active, dtype=wp.int32, device=device)
        # state_section_end_indices needed by the reset kernel.
        state_section_end_indices = wp.zeros(8, dtype=wp.int32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                reset_construction_buffers_kernel,
                dim=max(64, n_active, 8),
                inputs=[
                    construction_count, state_section_end_indices,
                    cid_to_partition_constraint_id, num_active_constraints,
                ],
                device=device,
            )
            wp.launch(
                record_all_interactions_kernel,
                dim=element_ids_by_color.shape[0],
                inputs=[
                    elements, interaction_id_to_partition,
                    element_ids_by_color, color_starts,
                    num_active_constraints,
                    wp.int32(8),  # num_bodies (large enough to avoid the particle branch)
                    wp.int32(max_partitions), wp.int32(batch_size),
                    construction_keys, construction_count, cid_to_partition_constraint_id,
                ],
                device=device,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize()

        pcid = cid_to_partition_constraint_id.numpy()
        # Regular cids -> 0. Overflow cids 3, 4, 5 at positions 0, 1, 2.
        # batch_size=1: pcid = j/1 = j. So overflow cids get 0, 1, 2.
        np.testing.assert_array_equal(
            pcid, np.array([0, 0, 0, 0, 1, 2], dtype=np.int32),
            err_msg=f"pcid mismatch with C# ParallelId convention: got {pcid}",
        )


# ---------------------------------------------------------------------------
# Test 2: record_all_interactions appends keys for all nodes in element.
# ---------------------------------------------------------------------------


@CUDA_ONLY
class TestRecordAllInteractions(unittest.TestCase):
    """C# ``RecordAllInteractionsKernel`` appends ``(node, constraintId)``
    for every non-(-1) entry in ``ElementInteractionData``. Newton's
    ``record_all_interactions_kernel`` mirrors this exactly: any
    ``node >= 0`` gets a key appended via ``atomic_add``, regardless
    of whether it's a rigid body or a cloth particle (in unified
    node space)."""

    def test_appends_key_per_node_in_each_active_cid(self):
        device = wp.get_preferred_device()
        # Two cids in regular partition 0:
        #   cid 0: bodies = [0, 1] (2 nodes)
        #   cid 1: bodies = [2, 3, 4] (3 nodes)
        # Total expected keys: 5 (all in partition 0 since regular).
        n_active = 2
        elements = _make_elements_array([(0, 1), (2, 3, 4)], capacity=n_active, device=device)
        interaction_id_to_partition = wp.array([0, 0], dtype=wp.int32, device=device)
        element_ids_by_color = wp.array([0, 1], dtype=wp.int32, device=device)
        color_starts = wp.array([0, 2, 2], dtype=wp.int32, device=device)
        num_active_constraints = wp.array([n_active], dtype=wp.int32, device=device)
        construction_keys = wp.zeros(64, dtype=wp.int64, device=device)
        construction_count = wp.zeros(1, dtype=wp.int32, device=device)
        cid_to_pcid = wp.zeros(n_active, dtype=wp.int32, device=device)
        sse = wp.zeros(8, dtype=wp.int32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                reset_construction_buffers_kernel, dim=64,
                inputs=[construction_count, sse, cid_to_pcid, num_active_constraints],
                device=device,
            )
            wp.launch(
                record_all_interactions_kernel,
                dim=element_ids_by_color.shape[0],
                inputs=[
                    elements, interaction_id_to_partition,
                    element_ids_by_color, color_starts, num_active_constraints,
                    wp.int32(8), wp.int32(1), wp.int32(1),
                    construction_keys, construction_count, cid_to_pcid,
                ],
                device=device,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize()

        n = int(construction_count.numpy()[0])
        self.assertEqual(n, 5, f"expected 5 keys, got {n}")

        keys = construction_keys.numpy()[:n]
        # Each key = (node << 32) | partition (= 0 for regular).
        decoded = sorted(int(k >> 32) for k in keys)
        self.assertEqual(decoded, [0, 1, 2, 3, 4])

        # All partition fields should be 0 for regular partition.
        partitions = [int(k & 0xFFFFFFFF) for k in keys]
        self.assertEqual(partitions, [0] * 5)


# ---------------------------------------------------------------------------
# Test 3: sort + dedup produces unique sorted keys.
# ---------------------------------------------------------------------------


@CUDA_ONLY
class TestSortAndDedup(unittest.TestCase):
    """The full sort + ``mark_unique_keys_kernel`` + scan + compact
    pipeline produces unique keys in sorted order, mirroring the C#
    ``DuplicateFreeList.Sort + RemoveDuplicates`` chain."""

    def test_sort_dedup_returns_unique_sorted_keys(self):
        device = wp.get_preferred_device()
        from newton._src.solvers.phoenx.helpers.scan_and_sort import (  # noqa: PLC0415
            scan_variable_length, sort_variable_length_int64,
        )

        # 8 raw entries with 3 unique keys: (3,0), (1,0), (3,0), (5,1), (1,0), (3,0), (5,1), (1,0).
        # Expected unique sorted: (1,0), (3,0), (5,1).
        raw = [(3, 0), (1, 0), (3, 0), (5, 1), (1, 0), (3, 0), (5, 1), (1, 0)]
        max_interactions = 32
        keys = wp.zeros(2 * max_interactions, dtype=wp.int64, device=device)
        values = wp.zeros(2 * max_interactions, dtype=wp.int32, device=device)
        # Populate keys[0..n] then count.

        @wp.kernel(enable_backward=False)
        def _populate_keys(keys: wp.array[wp.int64], packed: wp.array[wp.int64], n: wp.int32):
            i = wp.tid()
            if i < n:
                keys[i] = packed[i]

        packed_np = np.array([(b << 32) | p for (b, p) in raw], dtype=np.int64)
        packed = wp.array(packed_np, dtype=wp.int64, device=device)
        n = len(raw)
        construction_count = wp.array([n], dtype=wp.int32, device=device)
        wp.launch(_populate_keys, dim=n, inputs=[keys, packed, wp.int32(n)], device=device)

        # Sort then dedup.
        unique_flags = wp.zeros(2 * max_interactions, dtype=wp.int32, device=device)
        unique_offsets = wp.zeros(2 * max_interactions, dtype=wp.int32, device=device)
        unique_keys = wp.zeros(max_interactions, dtype=wp.int64, device=device)
        unique_count = wp.zeros(1, dtype=wp.int32, device=device)

        with wp.ScopedCapture(device=device) as capture:
            sort_variable_length_int64(keys, values, construction_count)
            wp.launch(
                mark_unique_keys_kernel, dim=2 * max_interactions,
                inputs=[keys, construction_count, unique_flags], device=device,
            )
            wp.copy(unique_offsets, unique_flags)
            scan_variable_length(unique_offsets, construction_count, inclusive=True)
            wp.launch(
                compact_unique_keys_kernel, dim=2 * max_interactions,
                inputs=[
                    keys, unique_flags, unique_offsets, construction_count,
                    unique_keys, unique_count,
                ],
                device=device,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize()

        n_uniq = int(unique_count.numpy()[0])
        self.assertEqual(n_uniq, 3, f"expected 3 unique keys, got {n_uniq}")
        out = unique_keys.numpy()[:n_uniq]
        decoded = sorted([(int(k >> 32), int(k & 0xFFFFFFFF)) for k in out])
        self.assertEqual(decoded, [(1, 0), (3, 0), (5, 1)])


# ---------------------------------------------------------------------------
# Test 4: build_partition_list + prefix_max_scan produces the C# CSR.
# ---------------------------------------------------------------------------


@CUDA_ONLY
class TestBuildPartitionListAndPrefixMax(unittest.TestCase):
    """``BuildInteractionGraphBuildKernel`` walks unique keys (sorted
    by ``(rigid, partition)``) and writes ``partition_list[k] =
    partition`` plus ``state_section_end_indices[rigid] = k+1`` at
    body boundaries. The C# ``maxScan.inclusiveScan`` then fills
    bodies with no entries by carrying the running max forward.
    Newton's pair (``build_partition_list_kernel`` +
    ``prefix_max_scan_kernel``) mirrors this exactly.
    """

    def test_csr_layout_matches_csharp_after_prefix_max(self):
        device = wp.get_preferred_device()
        # Unique keys after sort: (0,0), (0,1), (1,0), (3,0), (3,2).
        # Body 2 has NO entries -> prefix-max should carry body 1's
        # end (3) forward.
        unique_pairs = [(0, 0), (0, 1), (1, 0), (3, 0), (3, 2)]
        unique_keys = wp.array(
            np.array([(b << 32) | p for (b, p) in unique_pairs], dtype=np.int64),
            dtype=wp.int64, device=device,
        )
        unique_count = wp.array([len(unique_pairs)], dtype=wp.int32, device=device)
        partition_list = wp.zeros(16, dtype=wp.int32, device=device)
        state_section_end_indices = wp.zeros(4, dtype=wp.int32, device=device)
        # Reset section_end_indices to 0 first.
        state_section_end_indices.zero_()

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                build_partition_list_kernel, dim=16,
                inputs=[unique_keys, unique_count, partition_list, state_section_end_indices],
                device=device,
            )
            wp.launch(
                prefix_max_scan_kernel, dim=1,
                inputs=[state_section_end_indices, wp.int32(4)],
                device=device,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize()

        pl = partition_list.numpy()[:5]
        np.testing.assert_array_equal(
            pl, np.array([0, 1, 0, 0, 2], dtype=np.int32),
            err_msg=f"partition_list mismatch: {pl}",
        )
        sse = state_section_end_indices.numpy()
        # Pre-prefix-max: body 0 -> 2 (last entry at k=1, +1=2),
        # body 1 -> 3 (last entry at k=2, +1=3), body 2 -> 0 (no entries),
        # body 3 -> 5 (last entry at k=4, +1=5).
        # After prefix-max: [2, 3, 3, 5].
        np.testing.assert_array_equal(
            sse, np.array([2, 3, 3, 5], dtype=np.int32),
            err_msg=f"state_section_end_indices mismatch: {sse}",
        )


# ---------------------------------------------------------------------------
# Test 5: read_state lookup correctness.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _drive_read_state(
    graph: InteractionGraphData,
    out_idx: wp.array[wp.int32],
    out_inv_factor: wp.array[wp.int32],
    queries_pcid: wp.array[wp.int32],
    queries_body: wp.array[wp.int32],
):
    i = wp.tid()
    if i >= queries_pcid.shape[0]:
        return
    pcid = queries_pcid[i]
    body = queries_body[i]
    # Static-fallback path: pass dummy body-store args so the
    # function returns finite state if it falls back.
    state, inv_factor, idx = read_state(
        graph, pcid, body,
        wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)),
        wp.quatf(0.0, 0.0, 0.0, 1.0),
        wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)),
        wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)),
        wp.int32(ACCESS_MODE_VELOCITY_LEVEL),
        wp.float32(60.0),
    )
    out_idx[i] = idx
    out_inv_factor[i] = inv_factor


@CUDA_ONLY
class TestReadState(unittest.TestCase):
    """C# ``ConstraintHelper.cuh:153-169`` ``ReadState``:

    * registered body: returns ``(state, inv_factor=count, state_index)``
    * unregistered body in range: ``(synthesised_static, 0, -1)``
    * out-of-range body: ``(synthesised_static, 0, -1)``
    """

    def test_registered_unregistered_and_out_of_range(self):
        device = wp.get_preferred_device()
        # Body 0: 2 entries (partitions 0, 3).
        # Body 1: 1 entry (partition 5).
        # Body 2: NO entries (in-range but not registered).
        graph = _make_graph_with_entries(
            [(0, 0), (0, 3), (1, 5)],
            max_rigid_bodies=4, max_interactions=8, device=device,
        )
        # Queries:
        #   (pcid=0, body=0) -> idx=0, inv_factor=2 (body 0 has 2 entries)
        #   (pcid=3, body=0) -> idx=1, inv_factor=2
        #   (pcid=99, body=0) -> idx=-1, inv_factor=0 (pcid not in body 0's section)
        #   (pcid=5, body=1) -> idx>=0, inv_factor=1
        #   (pcid=0, body=2) -> idx=-1, inv_factor=0 (body 2 has no entries)
        #   (pcid=0, body=99) -> idx=-1, inv_factor=0 (body 99 out of range)
        queries_pcid = wp.array([0, 3, 99, 5, 0, 0], dtype=wp.int32, device=device)
        queries_body = wp.array([0, 0, 0, 1, 2, 99], dtype=wp.int32, device=device)
        out_idx = wp.zeros(6, dtype=wp.int32, device=device)
        out_inv_factor = wp.zeros(6, dtype=wp.int32, device=device)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                _drive_read_state, dim=6,
                inputs=[graph.data, out_idx, out_inv_factor, queries_pcid, queries_body],
                device=device,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize()

        idx = out_idx.numpy()
        inv_factor = out_inv_factor.numpy()
        # Body 0's section: [0,3] sorted -> partition_list = [0, 3]. So pcid=0 -> idx=0, pcid=3 -> idx=1.
        self.assertEqual(idx[0], 0)
        self.assertEqual(inv_factor[0], 2)
        self.assertEqual(idx[1], 1)
        self.assertEqual(inv_factor[1], 2)
        # pcid=99 not in body 0's section -> idx=-1
        self.assertEqual(idx[2], -1)
        self.assertEqual(inv_factor[2], 0)
        # body 1 has pcid=5 -> registered.
        self.assertGreaterEqual(idx[3], 0)
        self.assertEqual(inv_factor[3], 1)
        # body 2 in range but no entries -> idx=-1
        self.assertEqual(idx[4], -1)
        self.assertEqual(inv_factor[4], 0)
        # body 99 out of range -> idx=-1
        self.assertEqual(idx[5], -1)
        self.assertEqual(inv_factor[5], 0)


# ---------------------------------------------------------------------------
# Test 6: broadcast unified handles bodies + particles.
# ---------------------------------------------------------------------------


@CUDA_ONLY
class TestBroadcastUnified(unittest.TestCase):
    """``broadcast_to_copy_states_unified_kernel`` initialises every
    copy state from the body store (rigid path) or particle store
    (particle path). C# pattern: copies of node n hold
    ``TinyRigidState(velocity, angular_velocity, predicted position,
    predicted orientation)`` after the call. Particles use identity
    orientation and zero angular velocity.
    """

    def test_rigid_and_particle_copies_initialised_from_stores(self):
        device = wp.get_preferred_device()
        num_bodies = 2
        num_particles = 1
        num_nodes = num_bodies + num_particles
        # Body 0: in 1 partition. Body 1: in 2 partitions (overflow simulation).
        # Particle 0 (node id = num_bodies + 0 = 2): in 1 partition.
        graph = _make_graph_with_entries(
            [(0, 0), (1, 0), (1, 1), (2, 0)],
            max_rigid_bodies=num_nodes, max_interactions=8, device=device,
        )
        bodies = body_container_zeros(num_bodies, device=device)
        bodies.position.assign(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
        bodies.orientation.assign(np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        bodies.velocity.assign(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32))
        bodies.angular_velocity.assign(np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32))
        particles = particle_container_zeros(num_particles, device=device)
        particles.position.assign(np.array([[5.0, 0.0, 0.0]], dtype=np.float32))
        particles.velocity.assign(np.array([[0.0, 0.0, -3.0]], dtype=np.float32))
        dt = 1.0 / 60.0
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                broadcast_to_copy_states_unified_kernel,
                dim=num_nodes,
                inputs=[
                    graph.data,
                    bodies.position, bodies.orientation,
                    bodies.velocity, bodies.angular_velocity,
                    particles, wp.int32(num_bodies), wp.float32(dt),
                ],
                device=device,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize()

        ts = graph._tiny_states.numpy()
        # Body 0: section [0, 1) -> 1 copy.
        # Body 1: section [1, 3) -> 2 copies (one each for partition 0, 1).
        # Particle 0: section [3, 4) -> 1 copy.
        # Body 0 copy: velocity = (1, 0, 0).
        np.testing.assert_allclose(ts[0]["velocity"], [1.0, 0.0, 0.0], rtol=1e-5)
        np.testing.assert_allclose(ts[0]["angular_velocity"], [0.0, 0.0, 0.5], rtol=1e-5)
        # Body 1 copies: both have velocity = (0, 2, 0).
        np.testing.assert_allclose(ts[1]["velocity"], [0.0, 2.0, 0.0], rtol=1e-5)
        np.testing.assert_allclose(ts[2]["velocity"], [0.0, 2.0, 0.0], rtol=1e-5)
        # Particle 0 copy: velocity = (0, 0, -3), angular velocity = 0,
        # orientation = identity.
        np.testing.assert_allclose(ts[3]["velocity"], [0.0, 0.0, -3.0], rtol=1e-5)
        np.testing.assert_allclose(ts[3]["angular_velocity"], [0.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(ts[3]["orientation"], [0.0, 0.0, 0.0, 1.0], atol=1e-7)


# ---------------------------------------------------------------------------
# Test 7: average_and_broadcast averages copies.
# ---------------------------------------------------------------------------


@CUDA_ONLY
class TestAverageAndBroadcast(unittest.TestCase):
    """``average_and_broadcast_unified_kernel`` averages a body's
    copies and writes the mean to all copies. C#
    ``MassSplittingTypes.cuh:206-245``."""

    def test_three_copies_average_to_mean(self):
        device = wp.get_preferred_device()
        num_bodies = 1
        num_nodes = 1
        # Body 0 with 3 entries -> 3 copies.
        graph = _make_graph_with_entries(
            [(0, 0), (0, 1), (0, 2)],
            max_rigid_bodies=num_nodes, max_interactions=8, device=device,
        )
        # Manually set each copy to a different velocity; average kernel
        # should make them all equal to the mean.
        ts_np = graph._tiny_states.numpy().copy()
        ts_np[0]["velocity"] = (1.0, 0.0, 0.0)
        ts_np[0]["angular_velocity"] = (0.0, 0.0, 0.0)
        ts_np[0]["orientation"] = (0.0, 0.0, 0.0, 1.0)
        ts_np[0]["access_mode"] = ACCESS_MODE_VELOCITY_LEVEL
        ts_np[1]["velocity"] = (3.0, 0.0, 0.0)
        ts_np[1]["angular_velocity"] = (0.0, 0.0, 0.0)
        ts_np[1]["orientation"] = (0.0, 0.0, 0.0, 1.0)
        ts_np[1]["access_mode"] = ACCESS_MODE_VELOCITY_LEVEL
        ts_np[2]["velocity"] = (5.0, 0.0, 0.0)
        ts_np[2]["angular_velocity"] = (0.0, 0.0, 0.0)
        ts_np[2]["orientation"] = (0.0, 0.0, 0.0, 1.0)
        ts_np[2]["access_mode"] = ACCESS_MODE_VELOCITY_LEVEL
        graph._tiny_states.assign(ts_np)

        bodies = body_container_zeros(num_bodies, device=device)
        bodies.orientation.assign(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        particles = particle_container_zeros(1, device=device)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                average_and_broadcast_unified_kernel,
                dim=num_nodes,
                inputs=[
                    graph.data,
                    bodies.position, bodies.orientation,
                    particles, wp.int32(num_bodies), wp.float32(60.0),
                ],
                device=device,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize()

        ts = graph._tiny_states.numpy()
        # All 3 copies should have velocity = (3, 0, 0) (mean of 1, 3, 5).
        for i in range(3):
            np.testing.assert_allclose(
                ts[i]["velocity"], [3.0, 0.0, 0.0], rtol=1e-5,
                err_msg=f"copy {i} velocity != mean: {ts[i]['velocity']}",
            )


# ---------------------------------------------------------------------------
# Test 8: write_back commits to body / particle store.
# ---------------------------------------------------------------------------


@CUDA_ONLY
class TestWriteBackUnified(unittest.TestCase):
    """``copy_state_into_unified_kernel`` writes the slot-0 copy's
    velocity / angular velocity back into the body or particle
    store. C# ``MassSplittingTypes.cuh:282-300``."""

    def test_write_back_to_body_and_particle(self):
        device = wp.get_preferred_device()
        num_bodies = 1
        num_particles = 1
        num_nodes = num_bodies + num_particles
        graph = _make_graph_with_entries(
            [(0, 0), (1, 0)],
            max_rigid_bodies=num_nodes, max_interactions=8, device=device,
        )
        ts_np = graph._tiny_states.numpy().copy()
        ts_np[0]["velocity"] = (7.0, 0.0, 0.0)
        ts_np[0]["angular_velocity"] = (0.0, 0.0, 1.5)
        ts_np[0]["orientation"] = (0.0, 0.0, 0.0, 1.0)
        ts_np[0]["access_mode"] = ACCESS_MODE_VELOCITY_LEVEL
        ts_np[1]["velocity"] = (-2.0, 0.0, 0.0)
        ts_np[1]["angular_velocity"] = (0.0, 0.0, 0.0)
        ts_np[1]["orientation"] = (0.0, 0.0, 0.0, 1.0)
        ts_np[1]["access_mode"] = ACCESS_MODE_VELOCITY_LEVEL
        graph._tiny_states.assign(ts_np)

        bodies = body_container_zeros(num_bodies, device=device)
        bodies.orientation.assign(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        particles = particle_container_zeros(num_particles, device=device)
        particles.position.assign(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                copy_state_into_unified_kernel,
                dim=num_nodes,
                inputs=[
                    graph.data,
                    bodies.position, bodies.orientation,
                    bodies.velocity, bodies.angular_velocity,
                    particles, wp.int32(num_bodies), wp.float32(60.0),
                ],
                device=device,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize()

        body_v = bodies.velocity.numpy()
        body_w = bodies.angular_velocity.numpy()
        np.testing.assert_allclose(body_v[0], [7.0, 0.0, 0.0], rtol=1e-5)
        np.testing.assert_allclose(body_w[0], [0.0, 0.0, 1.5], rtol=1e-5)
        particle_v = particles.velocity.numpy()
        np.testing.assert_allclose(particle_v[0], [-2.0, 0.0, 0.0], rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 9: write_state respects idx=-1 (no-op for unregistered).
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _drive_write_state(
    graph: InteractionGraphData,
    state_input: TinyRigidState,
    idx: wp.int32,
):
    write_state(graph, idx, state_input)


@CUDA_ONLY
class TestWriteState(unittest.TestCase):
    """``write_state(graph, idx, state)`` writes when idx >= 0,
    no-ops when idx < 0. C# ``ConstraintHelper.cuh:214-233``."""

    def test_write_at_valid_idx(self):
        device = wp.get_preferred_device()
        graph = _make_graph_with_entries(
            [(0, 0), (1, 0)], max_rigid_bodies=2, max_interactions=4, device=device,
        )
        s = TinyRigidState()
        s.velocity = wp.vec3f(11.0, 22.0, 33.0)
        s.angular_velocity = wp.vec3f(0.0, 0.0, 0.0)
        s.position = wp.vec3f(0.0, 0.0, 0.0)
        s.orientation = wp.quatf(0.0, 0.0, 0.0, 1.0)
        s.access_mode = wp.int32(ACCESS_MODE_VELOCITY_LEVEL)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(_drive_write_state, dim=1, inputs=[graph.data, s, wp.int32(0)], device=device)
        wp.capture_launch(capture.graph)
        wp.synchronize()
        ts = graph._tiny_states.numpy()
        np.testing.assert_allclose(ts[0]["velocity"], [11.0, 22.0, 33.0], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
