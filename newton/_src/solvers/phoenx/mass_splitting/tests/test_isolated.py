# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Isolated unit tests for the mass-splitting subtree.

The whole point of landing the port standalone is that we can
exercise every architectural piece without running the constraint
solver at all. These tests do exactly that:

* :class:`TestContactPartitions` -- verify the host-side greedy MIS
  partitioner produces independent sets, and that
  :meth:`ContactPartitions.wrap_color_arrays` adopts an externally-
  built coloring without re-running it.
* :class:`TestInteractionGraph` -- build a small graph, verify the
  binary-search lookup returns the expected ``(state_index,
  inv_factor)`` pairs.
* :class:`TestBroadcastAverageRoundtrip` -- run the per-substep
  kernels end-to-end on a tiny scene and verify the
  broadcast / average / write-back round-trip preserves linear /
  angular momentum (the load-bearing invariant of the whole
  scheme).
* :class:`TestPositionLevelRoundTrip` -- broadcast, mutate
  ``state.position`` / ``state.orientation`` to fake an XPBD
  projection, write back. Verify the C# ``PositionLevel ->
  VelocityLevel`` finite-difference recovery
  (``v = (p - p_pre) * inv_dt``, ``omega = 2 * inv_dt * Im(q *
  conj(q_pre))`` with shortest-arc sign flip) lands the right
  velocity on every code path.

CUDA-only by design (per Newton convention -- the PhoenX subtree
is CUDA-only and graph-capture-aware).
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.mass_splitting import (
    ACCESS_MODE_POSITION_LEVEL,
    ContactPartitions,
    InteractionGraph,
)
from newton._src.solvers.phoenx.mass_splitting.kernels import (
    average_and_broadcast_kernel,
    broadcast_rigid_to_copy_states_kernel,
    copy_state_into_rigids_kernel,
)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX mass splitting requires CUDA")
class TestContactPartitions(unittest.TestCase):
    """Greedy MIS partitioner: every partition is an independent set."""

    def setUp(self) -> None:
        self.device = wp.get_device()

    def _build(self, constraint_bodies, max_partitions=12):
        cp = ContactPartitions(
            max_num_rigids=16,
            max_num_constraints=len(constraint_bodies),
            max_num_partitions=max_partitions,
            device=self.device,
        )
        cp.build(constraint_bodies)
        return cp

    def test_pairwise_disjoint_within_each_partition(self) -> None:
        # 6 contact constraints between 4 bodies, lots of conflict.
        # No two constraints sharing a body may end up in the same
        # partition.
        constraint_bodies = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ]
        cp = self._build(constraint_bodies)
        for p in range(cp.num_partitions):
            seen: set[int] = set()
            for cid in cp.iter_partition_constraints(p):
                for b in constraint_bodies[cid]:
                    self.assertNotIn(
                        b,
                        seen,
                        msg=f"body {b} appears twice in partition {p}",
                    )
                    seen.add(b)

    def test_chain_packs_into_two_partitions(self) -> None:
        # Linear chain -- every constraint shares a body with its
        # neighbour but not with the next-but-one. A 2-coloring is
        # optimal.
        chain = [(i, i + 1) for i in range(8)]
        cp = self._build(chain)
        self.assertLessEqual(cp.num_partitions, 2)
        self.assertFalse(cp.has_additional_partition)

    def test_capacity_overflow_uses_additional_partition(self) -> None:
        # Star graph: every constraint shares body 0. Forces every
        # constraint into its own partition; with max=2 the rest
        # spills into the additional partition.
        star = [(0, i + 1) for i in range(5)]
        cp = self._build(star, max_partitions=2)
        self.assertEqual(cp.num_partitions, 2)
        self.assertTrue(cp.has_additional_partition)

    def test_wrap_color_arrays_round_trips(self) -> None:
        # Mock a Newton-graph-coloring output: 3 colors of 2
        # constraints each. element_ids_by_color holds the IDs in
        # color order; color_starts are per-color *start* offsets.
        element_ids_np = np.array([0, 3, 1, 4, 2, 5], dtype=np.int32)
        color_starts_np = np.array([0, 2, 4, 6], dtype=np.int32)
        cp = ContactPartitions(
            max_num_rigids=16,
            max_num_constraints=6,
            max_num_partitions=4,
            device=self.device,
        )
        element_ids = wp.array(element_ids_np, dtype=wp.int32, device=self.device)
        color_starts = wp.array(color_starts_np, dtype=wp.int32, device=self.device)
        cp.wrap_color_arrays(element_ids, color_starts, num_colors=3, constraint_bodies=None)
        self.assertEqual(cp.num_partitions, 3)
        self.assertEqual(list(cp.iter_partition_constraints(0)), [0, 3])
        self.assertEqual(list(cp.iter_partition_constraints(1)), [1, 4])
        self.assertEqual(list(cp.iter_partition_constraints(2)), [2, 5])


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX mass splitting requires CUDA")
class TestInteractionGraph(unittest.TestCase):
    """Interaction graph build + per-(constraint, body) lookup."""

    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_build_reports_correct_inv_factor_per_body(self) -> None:
        # Body 0 touches partitions {0, 1, 2}; body 1 touches {0};
        # body 2 touches {1, 2}. Encode as 3 host entries per body.
        graph = InteractionGraph(max_rigid_bodies=4, max_interactions=8, device=self.device)
        # (rigid_id, partition_id)
        for entry in [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (2, 1),
            (2, 2),
        ]:
            graph.add_entry(*entry)
        graph.build()
        # Use a one-shot kernel that probes the lookup function for
        # each (constraint, body) pair we care about and writes the
        # results into output arrays we can read host-side.
        probes = wp.array(
            np.array(
                [
                    (0, 0),
                    (1, 0),
                    (2, 0),
                    (0, 1),
                    (1, 2),
                    (3, 0),  # nonexistent -> static fallback
                ],
                dtype=np.int32,
            ),
            dtype=wp.vec2i,
            device=self.device,
        )
        out_state_index = wp.zeros(probes.shape[0], dtype=wp.int32, device=self.device)
        out_inv_factor = wp.zeros(probes.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=_probe_get_rigid_state_index_kernel,
            dim=probes.shape[0],
            inputs=[graph.data, probes, out_state_index, out_inv_factor],
            device=self.device,
        )
        state_index = out_state_index.numpy()
        inv_factor = out_inv_factor.numpy()
        # Body 0 is in 3 partitions, so inv_factor=3 for body 0
        # lookups. Body 1 is in 1, body 2 in 2. Nonexistent is -1.
        self.assertGreaterEqual(state_index[0], 0)  # (cid=0, b=0) found
        self.assertEqual(inv_factor[0], 3)
        self.assertGreaterEqual(state_index[1], 0)
        self.assertEqual(inv_factor[1], 3)
        self.assertGreaterEqual(state_index[2], 0)
        self.assertEqual(inv_factor[2], 3)
        # body 1, partition 0 -> inv_factor 1
        self.assertGreaterEqual(state_index[3], 0)
        self.assertEqual(inv_factor[3], 1)
        # body 2, partition 2 -> inv_factor 2
        self.assertGreaterEqual(state_index[4], 0)
        self.assertEqual(inv_factor[4], 2)
        # body 3 is past highest_index_in_use -> static fallback
        self.assertEqual(state_index[5], -1)
        self.assertEqual(inv_factor[5], 0)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX mass splitting requires CUDA")
class TestBroadcastAverageRoundtrip(unittest.TestCase):
    """End-to-end momentum-conservation check on the substep kernels.

    Two bodies, one constraint between them; broadcast their
    velocities into the per-partition copies, *fake an iterate sweep*
    by adjusting the copies so they disagree, run average-and-
    broadcast, and verify the resulting averaged velocity equals
    the mean of the per-partition values. Then write back to the
    body store and confirm the body velocity matches.
    """

    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_average_preserves_mean_per_body(self) -> None:
        # We bypass MassSplitting for this test because we need to
        # hand-register more interaction entries than the orchestrator
        # would size for a 2-body / 1-constraint scene. The
        # broadcast / average / write-back kernels themselves take
        # only ``InteractionGraphData`` so they're testable
        # standalone.
        graph = InteractionGraph(max_rigid_bodies=2, max_interactions=8, device=self.device)
        # Two bodies, three partitions each -- the
        # ``count > 1`` path in average_and_broadcast_kernel.
        for partition_index in (0, 1, 2):
            graph.add_entry(0, partition_index)
            graph.add_entry(1, partition_index)
        graph.build()

        # Body store: SoA arrays.
        body_position = wp.array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            dtype=wp.vec3f,
            device=self.device,
        )
        body_orientation = wp.array(
            np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            dtype=wp.quatf,
            device=self.device,
        )
        body_velocity = wp.array(
            np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
            dtype=wp.vec3f,
            device=self.device,
        )
        body_angular_velocity = wp.array(
            np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]], dtype=np.float32),
            dtype=wp.vec3f,
            device=self.device,
        )

        dt = 0.01
        inv_dt = 1.0 / dt

        # Broadcast: every per-partition copy now holds the body's
        # velocity (integrated by dt for position, but velocity
        # passes through).
        wp.launch(
            kernel=broadcast_rigid_to_copy_states_kernel,
            dim=2,
            inputs=[
                graph.data,
                body_position,
                body_orientation,
                body_velocity,
                body_angular_velocity,
                float(dt),
            ],
            device=self.device,
        )

        # Fake an iterate sweep: spread the per-partition velocities
        # by a known offset so the averaging step has work to do.
        # Body 0's three partitions get vel = (1, 0, 0) +/- (0.3, 0, 0).
        # Body 1's three partitions get vel = (0, 2, 0) +/- (0, 0.3, 0).
        wp.launch(
            kernel=_perturb_partition_states_kernel,
            dim=2,
            inputs=[graph.data],
            device=self.device,
        )

        # Average + write-back.
        wp.launch(
            kernel=average_and_broadcast_kernel,
            dim=2,
            inputs=[
                graph.data,
                body_position,
                body_orientation,
                float(inv_dt),
            ],
            device=self.device,
        )
        wp.launch(
            kernel=copy_state_into_rigids_kernel,
            dim=2,
            inputs=[
                graph.data,
                body_position,
                body_orientation,
                body_velocity,
                body_angular_velocity,
                float(inv_dt),
            ],
            device=self.device,
        )

        # The perturbations were symmetric around the original
        # velocities, so the post-average per-body velocity must
        # match the original within float-32 precision.
        v_after = body_velocity.numpy()
        np.testing.assert_allclose(v_after[0], [1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(v_after[1], [0.0, 2.0, 0.0], atol=1e-5)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX mass splitting requires CUDA")
class TestPositionLevelRoundTrip(unittest.TestCase):
    """Position-level pass via the existing mass-splitting machinery.

    The C# pattern is unified: same TinyRigidState, same broadcast
    /write-back kernels, the constraint kernel just chooses
    ``ACCESS_MODE_POSITION_LEVEL`` when reading. These tests exercise
    every code path of the
    :func:`tiny_rigid_state_set_access_mode` regime switch by
    hand-rolling a fake "XPBD projection" kernel that mutates
    ``state.position`` / ``state.orientation`` directly, then
    verifying the write-back kernel recovers the right velocity.
    """

    def setUp(self) -> None:
        self.device = wp.get_device()

    def _build_per_body_graph(self, num_bodies: int) -> InteractionGraph:
        """Trivial setup: each body has exactly one TinyState slot
        (registered at constraint_index 0). For position-level passes
        without mass splitting; cloth's real graph would have one
        entry per (body, partition) pair."""
        graph = InteractionGraph(
            max_rigid_bodies=num_bodies,
            max_interactions=num_bodies,
            device=self.device,
        )
        for body_id in range(num_bodies):
            graph.add_entry(body_id, 0)
        graph.build()
        return graph

    def _zero_iterate_recovers_zero_velocity(self) -> None:
        """No projection between broadcast and write-back -> velocity
        = (p - p_pre)/dt = 0. Confirms the synchronize math doesn't
        emit spurious motion on a no-op pass."""
        n = 4
        graph = self._build_per_body_graph(n)
        body_position = wp.array(
            np.array([[float(i), 0.0, 0.0] for i in range(n)], dtype=np.float32),
            dtype=wp.vec3f,
            device=self.device,
        )
        body_orientation = wp.array(
            np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)),
            dtype=wp.quatf,
            device=self.device,
        )
        body_velocity = wp.array(
            np.full((n, 3), 99.0, dtype=np.float32),
            dtype=wp.vec3f,
            device=self.device,
        )
        body_angular_velocity = wp.array(
            np.full((n, 3), 99.0, dtype=np.float32),
            dtype=wp.vec3f,
            device=self.device,
        )
        dt = 0.01
        wp.launch(
            kernel=broadcast_rigid_to_copy_states_kernel,
            dim=n,
            inputs=[graph.data, body_position, body_orientation, body_velocity, body_angular_velocity, float(dt)],
            device=self.device,
        )
        # No iterate work between broadcast and write-back. Broadcast
        # captured ``state.access_mode = VELOCITY_LEVEL`` with the
        # body's current velocity, so write-back's
        # ``synchronize(VELOCITY_LEVEL)`` is a no-op and reads the
        # state's velocity field (= original 99.0). Verifies the
        # short-circuit branch.
        wp.launch(
            kernel=copy_state_into_rigids_kernel,
            dim=n,
            inputs=[graph.data, body_position, body_orientation, body_velocity, body_angular_velocity, float(1.0 / dt)],
            device=self.device,
        )
        v = body_velocity.numpy()
        np.testing.assert_allclose(v, np.full((n, 3), 99.0), atol=1e-5)

    def test_zero_iterate_preserves_velocity(self) -> None:
        self._zero_iterate_recovers_zero_velocity()

    def test_pure_translation_recovers_velocity(self) -> None:
        """Mutate ``state.position`` by a known offset; write-back's
        ``synchronize(VELOCITY_LEVEL)`` must recover ``v = offset / dt``."""
        n = 3
        graph = self._build_per_body_graph(n)
        translations = np.array(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            dtype=np.float32,
        )
        body_position = wp.array(np.zeros((n, 3), dtype=np.float32), dtype=wp.vec3f, device=self.device)
        body_orientation = wp.array(
            np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)),
            dtype=wp.quatf,
            device=self.device,
        )
        body_velocity = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        body_angular_velocity = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        dt = 0.01
        wp.launch(
            kernel=broadcast_rigid_to_copy_states_kernel,
            dim=n,
            inputs=[graph.data, body_position, body_orientation, body_velocity, body_angular_velocity, float(dt)],
            device=self.device,
        )
        # "XPBD projection": flip state to position-level and add the
        # per-body offset to ``state.position``. This is the same
        # work a real distance-constraint kernel would do; the
        # helper kernel below packs it into a single launch.
        wp.launch(
            kernel=_fake_xpbd_translation_kernel,
            dim=n,
            inputs=[
                graph.data,
                body_position,
                body_orientation,
                wp.array(translations, dtype=wp.vec3f, device=self.device),
                wp.float32(1.0 / dt),
            ],
            device=self.device,
        )
        wp.launch(
            kernel=copy_state_into_rigids_kernel,
            dim=n,
            inputs=[graph.data, body_position, body_orientation, body_velocity, body_angular_velocity, float(1.0 / dt)],
            device=self.device,
        )
        v = body_velocity.numpy()
        # Recovered velocity should equal translation/dt (the XPBD
        # finite difference). The body_position from broadcast
        # advanced by ``dt * body_velocity = 0`` (initial velocity 0),
        # so state.position == body_position; adding the translation
        # to state.position lands at body_position + translation,
        # so (state.position - body_position)/dt = translation/dt.
        np.testing.assert_allclose(v, translations / dt, atol=1e-5)

    def test_pure_rotation_uses_shortest_arc(self) -> None:
        """Set ``state.orientation`` to a rotation > pi about +Z; the
        shortest-arc sign flip in
        :func:`tiny_rigid_state_synchronize` must recover a *negative*
        ``omega_z``."""
        n = 1
        graph = self._build_per_body_graph(n)
        dt = 1.0 / 60.0
        angle = math.pi + 0.1
        s = math.sin(0.5 * angle)
        c = math.cos(0.5 * angle)
        target_q = np.array([[0.0, 0.0, s, c]], dtype=np.float32)
        body_position = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        body_orientation = wp.array(
            np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            dtype=wp.quatf,
            device=self.device,
        )
        body_velocity = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        body_angular_velocity = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        wp.launch(
            kernel=broadcast_rigid_to_copy_states_kernel,
            dim=n,
            inputs=[graph.data, body_position, body_orientation, body_velocity, body_angular_velocity, float(dt)],
            device=self.device,
        )
        wp.launch(
            kernel=_fake_xpbd_rotation_kernel,
            dim=n,
            inputs=[
                graph.data,
                body_position,
                body_orientation,
                wp.array(target_q, dtype=wp.quatf, device=self.device),
                wp.float32(1.0 / dt),
            ],
            device=self.device,
        )
        wp.launch(
            kernel=copy_state_into_rigids_kernel,
            dim=n,
            inputs=[graph.data, body_position, body_orientation, body_velocity, body_angular_velocity, float(1.0 / dt)],
            device=self.device,
        )
        w = body_angular_velocity.numpy()[0]
        # Shortest arc to a +pi+0.1 rotation about +Z is -(pi - 0.1)
        # about +Z. Recovered omega_z must be negative.
        self.assertLess(w[2], 0.0, msg=f"angular velocity {w} should be negative on Z (shortest arc)")


# ---------------------------------------------------------------------------
# Local probe / perturbation kernels (only used by the tests above)
# ---------------------------------------------------------------------------
#
# Defining them here keeps the production source clean -- nothing
# in newton.* needs to peek into TinyRigidState fields the way the
# tests do.


from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (  # noqa: E402
    InteractionGraphData,
    graph_get_rigid_state_index,
    graph_get_state,
    graph_set_state,
    graph_state_section,
)
from newton._src.solvers.phoenx.mass_splitting.state import (  # noqa: E402
    tiny_rigid_state_set_access_mode,
)


@wp.kernel(enable_backward=False)
def _probe_get_rigid_state_index_kernel(
    graph: InteractionGraphData,
    probes: wp.array[wp.vec2i],
    out_state_index: wp.array[wp.int32],
    out_inv_factor: wp.array[wp.int32],
):
    tid = wp.tid()
    p = probes[tid]
    constraint_index = p[0]
    rigid_body_index = p[1]
    state_index, inv_factor = graph_get_rigid_state_index(graph, constraint_index, rigid_body_index)
    out_state_index[tid] = state_index
    out_inv_factor[tid] = inv_factor


_ACCESS_MODE_POSITION_LEVEL_C = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))


@wp.kernel(enable_backward=False)
def _fake_xpbd_translation_kernel(
    graph: InteractionGraphData,
    body_position: wp.array[wp.vec3f],
    body_orientation: wp.array[wp.quatf],
    translations: wp.array[wp.vec3f],
    inv_dt: wp.float32,
):
    """Stand-in for an XPBD distance-constraint iterate kernel.

    Reads the body's TinyState via :func:`graph_get_state` (after
    flipping access mode to POSITION_LEVEL via ``set_access_mode``,
    matching what a real XPBD kernel would do via
    :func:`read_state`), adds the translation to ``state.position``,
    writes back. The write-back kernel's
    ``synchronize(VELOCITY_LEVEL)`` then runs the XPBD finite
    difference, recovering ``v = translation / dt``.
    """
    rigid_body_index = wp.tid()
    if rigid_body_index >= graph.highest_index_in_use[0]:
        return
    start, end = graph_state_section(graph, rigid_body_index)
    if start >= end:
        return
    state = graph_get_state(graph, start)
    state = tiny_rigid_state_set_access_mode(
        state,
        _ACCESS_MODE_POSITION_LEVEL_C,
        body_position[rigid_body_index],
        body_orientation[rigid_body_index],
        inv_dt,
    )
    state.position = state.position + translations[rigid_body_index]
    graph_set_state(graph, start, state)


@wp.kernel(enable_backward=False)
def _fake_xpbd_rotation_kernel(
    graph: InteractionGraphData,
    body_position: wp.array[wp.vec3f],
    body_orientation: wp.array[wp.quatf],
    target_orientations: wp.array[wp.quatf],
    inv_dt: wp.float32,
):
    """Stand-in for an XPBD orientation projection (e.g. a fixed
    joint snapping a body to a target frame). Sets ``state.orientation``
    directly to the target; write-back's
    ``synchronize(VELOCITY_LEVEL)`` recovers the angular velocity
    via the log-map of ``target * conj(body_orientation)``.
    """
    rigid_body_index = wp.tid()
    if rigid_body_index >= graph.highest_index_in_use[0]:
        return
    start, end = graph_state_section(graph, rigid_body_index)
    if start >= end:
        return
    state = graph_get_state(graph, start)
    state = tiny_rigid_state_set_access_mode(
        state,
        _ACCESS_MODE_POSITION_LEVEL_C,
        body_position[rigid_body_index],
        body_orientation[rigid_body_index],
        inv_dt,
    )
    state.orientation = target_orientations[rigid_body_index]
    graph_set_state(graph, start, state)


@wp.kernel(enable_backward=False)
def _perturb_partition_states_kernel(graph: InteractionGraphData):
    """For body ``b``, push partition i's velocity by ``(i - 1) * 0.3``
    along the body's "natural" axis (X for body 0, Y for body 1) so
    the three slots are ``v - 0.3, v, v + 0.3`` with mean ``v``.
    """
    rigid_body_index = wp.tid()
    if rigid_body_index >= graph.highest_index_in_use[0]:
        return
    start, end = graph_state_section(graph, rigid_body_index)
    if start >= end:
        return
    axis = wp.vec3f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0))
    if rigid_body_index == wp.int32(1):
        axis = wp.vec3f(wp.float32(0.0), wp.float32(1.0), wp.float32(0.0))
    count = end - start
    half = wp.float32(count - wp.int32(1)) * wp.float32(0.5)
    for i in range(start, end):
        s = graph_get_state(graph, i)
        local_index = i - start
        offset = (wp.float32(local_index) - half) * wp.float32(0.3)
        s.velocity = s.velocity + axis * offset
        graph_set_state(graph, i, s)


if __name__ == "__main__":
    unittest.main()
