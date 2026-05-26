# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Broadcast / average / writeback round-trip and momentum-conservation
tests.

These pin down the contracts that Step 6 (substep loop integration)
will rely on:

* Broadcast fans body / particle state into every slot. After
  broadcast, every slot for a given node holds the same velocity /
  angular velocity (and the predicted position).
* AverageAndBroadcast over N slots collapses any per-slot velocity
  divergence back to the mean. Sum of slot velocities is invariant
  (momentum-conserving) when slot count > 1.
* CopyStateIntoRigids writes the slot-0 velocity back to body /
  particle. Round-trip with no inter-iter mutation must be the
  identity.
* All three kernels are no-ops when ``highest_index_in_use == 0``
  (mass splitting disabled).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_POSITION_LEVEL,
    ACCESS_MODE_STATIC,
    ACCESS_MODE_VELOCITY_LEVEL,
)
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.mass_splitting.copy_state import (
    copy_state_container_zeros,
)
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    build_interaction_graph,
    interaction_graph_scratch_zeros,
)
from newton._src.solvers.phoenx.mass_splitting.kernels import (
    launch_average_and_broadcast,
    launch_average_and_broadcast_grouped,
    launch_average_and_broadcast_rigid_velocity,
    launch_broadcast_rigid_to_copy_states,
    launch_copy_state_into_rigids,
)
from newton._src.solvers.phoenx.mass_splitting.tests.test_interaction_graph import (
    _seed_pairs_direct,
)
from newton._src.solvers.phoenx.particle import particle_container_zeros


def _setup_two_bodies_three_partitions(device):
    """Body 0: partitions {0, 1}. Body 1: partitions {0, 2, 5}.
    section_end = [2, 5]. 5 slots total."""
    capacity = 8
    num_bodies = 2
    num_particles = 1  # one particle so we exercise the particle branch too
    num_nodes = num_bodies + num_particles

    bodies = body_container_zeros(num_bodies=num_bodies, device=device)
    particles = particle_container_zeros(num_particles=num_particles, device=device)
    cs = copy_state_container_zeros(capacity=capacity, num_nodes=num_nodes, device=device)
    scratch = interaction_graph_scratch_zeros(capacity=capacity, device=device)

    # Seed body and particle state.
    bodies.position.assign(np.asarray([(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)], dtype=np.float32))
    bodies.orientation.assign(np.asarray([(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)], dtype=np.float32))
    bodies.velocity.assign(np.asarray([(0.5, 0.0, 0.0), (0.0, 0.7, 0.0)], dtype=np.float32))
    bodies.angular_velocity.assign(np.asarray([(0.1, 0.2, 0.3), (0.0, 0.0, 0.5)], dtype=np.float32))
    bodies.access_mode.assign(np.full(num_bodies, int(ACCESS_MODE_VELOCITY_LEVEL), dtype=np.int32))

    particles.position.assign(np.asarray([(5.0, 5.0, 5.0)], dtype=np.float32))
    particles.velocity.assign(np.asarray([(1.0, 0.0, -1.0)], dtype=np.float32))
    particles.access_mode.assign(np.asarray([int(ACCESS_MODE_VELOCITY_LEVEL)], dtype=np.int32))

    # Build the interaction graph: body 0 → {0, 1}; body 1 → {0, 2, 5};
    # particle (node 2) gets no entries.
    pairs = [(0, 0), (0, 1), (1, 0), (1, 2), (1, 5)]
    _seed_pairs_direct(scratch, pairs, device)
    build_interaction_graph(scratch, cs)
    wp.synchronize_device(device)

    return bodies, particles, cs, num_bodies, num_nodes


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX mass-splitting tests are CUDA-only per feedback_phoenx_tests_capture_only.",
)
class TestBroadcastAverage(unittest.TestCase):
    def test_broadcast_fills_every_slot(self):
        device = wp.get_preferred_device()
        bodies, particles, cs, num_bodies, _ = _setup_two_bodies_three_partitions(device)
        dt = 0.01

        # Warm-up + capture.
        launch_broadcast_rigid_to_copy_states(cs, bodies, particles, num_bodies, dt)
        with wp.ScopedCapture(device=device) as capture:
            launch_broadcast_rigid_to_copy_states(cs, bodies, particles, num_bodies, dt)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        # Slot section_end after build = [2, 5, 5] (particle has 0 slots).
        section_end = cs.section_end.numpy()
        np.testing.assert_array_equal(section_end, [2, 5, 5])

        vel = cs.velocity.numpy()
        ang = cs.angular_velocity.numpy()
        pos = cs.position.numpy()
        mode = cs.access_mode.numpy()
        body_vel_h = bodies.velocity.numpy()
        body_ang_h = bodies.angular_velocity.numpy()
        body_pos_h = bodies.position.numpy()

        # Body 0 (slots 0..1) must hold body 0's velocity and forward-integrated pos.
        for s in range(0, 2):
            np.testing.assert_allclose(vel[s], body_vel_h[0])
            np.testing.assert_allclose(ang[s], body_ang_h[0])
            np.testing.assert_allclose(pos[s], body_pos_h[0] + dt * body_vel_h[0])
            self.assertEqual(int(mode[s]), int(ACCESS_MODE_VELOCITY_LEVEL))
        # Body 1 (slots 2..4) gets body 1's velocity.
        for s in range(2, 5):
            np.testing.assert_allclose(vel[s], body_vel_h[1])
            np.testing.assert_allclose(ang[s], body_ang_h[1])
            np.testing.assert_allclose(pos[s], body_pos_h[1] + dt * body_vel_h[1])

    def test_average_collapses_divergent_velocities(self):
        # Manually scribble different velocities into each of body 1's 3
        # slots, then average. The result must be the mean, broadcast
        # to all 3 slots. Body 0's two slots (currently equal post-broadcast)
        # stay equal.
        device = wp.get_preferred_device()
        bodies, particles, cs, num_bodies, _ = _setup_two_bodies_three_partitions(device)
        dt = 0.01
        launch_broadcast_rigid_to_copy_states(cs, bodies, particles, num_bodies, dt)
        wp.synchronize_device(device)

        # Body 1 occupies slots [2, 5). Diverge them.
        vel_h = cs.velocity.numpy().copy()
        ang_h = cs.angular_velocity.numpy().copy()
        slot_v = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        slot_w = [
            (0.0, 0.0, 0.0),
            (0.0, 0.6, 0.0),
            (0.6, 0.0, 0.0),
        ]
        for idx, s in enumerate(range(2, 5)):
            vel_h[s] = slot_v[idx]
            ang_h[s] = slot_w[idx]
        cs.velocity.assign(vel_h)
        cs.angular_velocity.assign(ang_h)

        # Warm-up + capture + launch average.
        launch_average_and_broadcast(cs, bodies, particles, num_bodies, 1.0 / dt)
        with wp.ScopedCapture(device=device) as capture:
            launch_average_and_broadcast(cs, bodies, particles, num_bodies, 1.0 / dt)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        vel = cs.velocity.numpy()
        ang = cs.angular_velocity.numpy()
        expected_v = np.mean(np.array(slot_v, dtype=np.float32), axis=0)
        expected_w = np.mean(np.array(slot_w, dtype=np.float32), axis=0)
        for s in range(2, 5):
            np.testing.assert_allclose(vel[s], expected_v, rtol=1e-6)
            np.testing.assert_allclose(ang[s], expected_w, rtol=1e-6)

        # Body 0's slots (2 of them, same velocity post-broadcast) stay
        # equal — averaging a tied pair is the identity.
        np.testing.assert_allclose(vel[0], vel[1])
        np.testing.assert_allclose(ang[0], ang[1])

    def test_average_is_momentum_conserving(self):
        # Sum_over_slots(velocity) BEFORE averaging == sum_over_slots(velocity)
        # AFTER averaging. This is the Tonge invariant: the average step
        # is mass-conserving as long as we sum then re-broadcast.
        device = wp.get_preferred_device()
        bodies, particles, cs, num_bodies, _ = _setup_two_bodies_three_partitions(device)
        dt = 0.01
        launch_broadcast_rigid_to_copy_states(cs, bodies, particles, num_bodies, dt)
        wp.synchronize_device(device)

        # Scribble per-slot velocities so the section sums are non-trivial.
        rng = np.random.default_rng(seed=20260511)
        vel_h = cs.velocity.numpy().copy()
        ang_h = cs.angular_velocity.numpy().copy()
        vel_h[0:5] = rng.normal(size=(5, 3)).astype(np.float32)
        ang_h[0:5] = rng.normal(size=(5, 3)).astype(np.float32)
        cs.velocity.assign(vel_h)
        cs.angular_velocity.assign(ang_h)

        # Per-body sum BEFORE.
        sum_b0_v_before = vel_h[0:2].sum(axis=0)
        sum_b1_v_before = vel_h[2:5].sum(axis=0)
        sum_b0_w_before = ang_h[0:2].sum(axis=0)
        sum_b1_w_before = ang_h[2:5].sum(axis=0)

        launch_average_and_broadcast(cs, bodies, particles, num_bodies, 1.0 / dt)
        wp.synchronize_device(device)

        vel_after = cs.velocity.numpy()
        ang_after = cs.angular_velocity.numpy()
        sum_b0_v_after = vel_after[0:2].sum(axis=0)
        sum_b1_v_after = vel_after[2:5].sum(axis=0)
        sum_b0_w_after = ang_after[0:2].sum(axis=0)
        sum_b1_w_after = ang_after[2:5].sum(axis=0)

        np.testing.assert_allclose(sum_b0_v_after, sum_b0_v_before, rtol=1e-5)
        np.testing.assert_allclose(sum_b1_v_after, sum_b1_v_before, rtol=1e-5)
        np.testing.assert_allclose(sum_b0_w_after, sum_b0_w_before, rtol=1e-5)
        np.testing.assert_allclose(sum_b1_w_after, sum_b1_w_before, rtol=1e-5)

    def test_rigid_velocity_average_matches_scalar_velocity_slots(self):
        # Rigid-only scenes have no position-level writers, so the
        # specialized average can skip access-mode synchronization and
        # still match the generic kernel for velocity-level slots.
        device = wp.get_preferred_device()
        bodies, particles, cs_scalar, num_bodies, _ = _setup_two_bodies_three_partitions(device)
        _, _, cs_rigid, _, _ = _setup_two_bodies_three_partitions(device)
        dt = 0.01
        launch_broadcast_rigid_to_copy_states(cs_scalar, bodies, particles, num_bodies, dt)
        launch_broadcast_rigid_to_copy_states(cs_rigid, bodies, particles, num_bodies, dt)
        wp.synchronize_device(device)

        rng = np.random.default_rng(seed=20260526)
        vel_h = cs_scalar.velocity.numpy().copy()
        ang_h = cs_scalar.angular_velocity.numpy().copy()
        vel_h[0:5] = rng.normal(size=(5, 3)).astype(np.float32)
        ang_h[0:5] = rng.normal(size=(5, 3)).astype(np.float32)
        for cs in (cs_scalar, cs_rigid):
            cs.velocity.assign(vel_h)
            cs.angular_velocity.assign(ang_h)

        launch_average_and_broadcast(cs_scalar, bodies, particles, num_bodies, 1.0 / dt)
        launch_average_and_broadcast_rigid_velocity(cs_rigid, bodies, particles, num_bodies, 1.0 / dt)
        wp.synchronize_device(device)

        np.testing.assert_array_equal(cs_rigid.velocity.numpy()[:5], cs_scalar.velocity.numpy()[:5])
        np.testing.assert_array_equal(cs_rigid.angular_velocity.numpy()[:5], cs_scalar.angular_velocity.numpy()[:5])

    def test_grouped_average_matches_scalar(self):
        device = wp.get_preferred_device()
        capacity = 16
        num_bodies = 1
        num_particles = 2
        num_nodes = num_bodies + num_particles
        bodies = body_container_zeros(num_bodies=num_bodies, device=device)
        particles = particle_container_zeros(num_particles=num_particles, device=device)
        cs_scalar = copy_state_container_zeros(capacity=capacity, num_nodes=num_nodes, device=device)
        cs_grouped = copy_state_container_zeros(capacity=capacity, num_nodes=num_nodes, device=device)
        scratch = interaction_graph_scratch_zeros(capacity=capacity, device=device)

        pairs = [(0, 0), (0, 1), (0, 4), (1, 0), (1, 1), (1, 3), (2, 0), (2, 2)]
        _seed_pairs_direct(scratch, pairs, device)
        build_interaction_graph(scratch, cs_scalar)
        _seed_pairs_direct(scratch, pairs, device)
        build_interaction_graph(scratch, cs_grouped)

        bodies.position_prev_substep.assign(np.asarray([(0.0, 0.0, 0.0)], dtype=np.float32))
        bodies.orientation_prev_substep.assign(np.asarray([(0.0, 0.0, 0.0, 1.0)], dtype=np.float32))
        particles.position_prev_substep.assign(np.asarray([(1.0, 2.0, 3.0), (-1.0, 0.5, 2.0)], dtype=np.float32))

        pos = np.zeros((capacity, 3), dtype=np.float32)
        vel = np.zeros((capacity, 3), dtype=np.float32)
        ang = np.zeros((capacity, 3), dtype=np.float32)
        mode = np.full(capacity, int(ACCESS_MODE_VELOCITY_LEVEL), dtype=np.int32)

        pos[0] = np.array([0.01, 0.02, 0.03], dtype=np.float32)
        mode[0] = int(ACCESS_MODE_POSITION_LEVEL)
        vel[1] = np.array([0.5, -0.25, 0.75], dtype=np.float32)
        ang[1] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vel[2] = np.array([1.0, 0.0, -1.0], dtype=np.float32)
        ang[2] = np.array([0.3, 0.0, -0.1], dtype=np.float32)

        pos[3] = np.array([1.03, 2.00, 3.00], dtype=np.float32)
        mode[3] = int(ACCESS_MODE_POSITION_LEVEL)
        vel[4] = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        pos[5] = np.array([0.97, 2.06, 3.00], dtype=np.float32)
        mode[5] = int(ACCESS_MODE_POSITION_LEVEL)

        vel[6] = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        pos[7] = np.array([-0.98, 0.51, 2.04], dtype=np.float32)
        mode[7] = int(ACCESS_MODE_POSITION_LEVEL)

        for cs in (cs_scalar, cs_grouped):
            cs.position.assign(pos)
            cs.velocity.assign(vel)
            cs.angular_velocity.assign(ang)
            cs.access_mode.assign(mode)

        inv_dt = 100.0
        launch_average_and_broadcast(cs_scalar, bodies, particles, num_bodies, inv_dt)
        launch_average_and_broadcast_grouped(cs_grouped, bodies, particles, num_bodies, inv_dt)
        wp.synchronize_device(device)

        np.testing.assert_array_equal(cs_grouped.access_mode.numpy()[:8], cs_scalar.access_mode.numpy()[:8])
        np.testing.assert_array_equal(cs_grouped.velocity.numpy()[:8], cs_scalar.velocity.numpy()[:8])
        np.testing.assert_array_equal(cs_grouped.angular_velocity.numpy()[:3], cs_scalar.angular_velocity.numpy()[:3])

    def test_writeback_round_trip_identity(self):
        # Broadcast → no constraint mutation → writeback. The body
        # velocities must come back byte-identical (the slot's
        # velocity is the source).
        device = wp.get_preferred_device()
        bodies, particles, cs, num_bodies, _ = _setup_two_bodies_three_partitions(device)
        dt = 0.01

        v_orig = bodies.velocity.numpy().copy()
        w_orig = bodies.angular_velocity.numpy().copy()

        launch_broadcast_rigid_to_copy_states(cs, bodies, particles, num_bodies, dt)
        launch_copy_state_into_rigids(cs, bodies, particles, num_bodies, 1.0 / dt)
        wp.synchronize_device(device)

        np.testing.assert_allclose(bodies.velocity.numpy(), v_orig)
        np.testing.assert_allclose(bodies.angular_velocity.numpy(), w_orig)

    def test_disabled_path_no_op(self):
        # highest_index_in_use == 0 (no build / no pairs). Every kernel
        # must short-circuit and leave bodies / particles / slots alone.
        device = wp.get_preferred_device()
        num_bodies = 2
        num_particles = 1
        bodies = body_container_zeros(num_bodies=num_bodies, device=device)
        particles = particle_container_zeros(num_particles=num_particles, device=device)
        cs = copy_state_container_zeros(capacity=4, num_nodes=num_bodies + num_particles, device=device)

        bodies.velocity.assign(np.asarray([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], dtype=np.float32))
        bodies.angular_velocity.assign(np.asarray([(0.1, 0.0, 0.0), (0.0, 0.2, 0.0)], dtype=np.float32))
        v_orig = bodies.velocity.numpy().copy()
        w_orig = bodies.angular_velocity.numpy().copy()
        slots_v_orig = cs.velocity.numpy().copy()

        dt = 0.01
        launch_broadcast_rigid_to_copy_states(cs, bodies, particles, num_bodies, dt)
        launch_average_and_broadcast(cs, bodies, particles, num_bodies, 1.0 / dt)
        launch_copy_state_into_rigids(cs, bodies, particles, num_bodies, 1.0 / dt)
        wp.synchronize_device(device)

        # Bodies untouched, slots untouched (broadcast skipped because
        # section ranges are all empty).
        np.testing.assert_array_equal(bodies.velocity.numpy(), v_orig)
        np.testing.assert_array_equal(bodies.angular_velocity.numpy(), w_orig)
        np.testing.assert_array_equal(cs.velocity.numpy(), slots_v_orig)

    def test_static_body_is_skipped_on_writeback(self):
        # A body with ACCESS_MODE_STATIC must not have its velocity
        # overwritten by writeback, even after broadcast stamps
        # ACCESS_MODE_STATIC into its slots.
        device = wp.get_preferred_device()
        bodies, particles, cs, num_bodies, _ = _setup_two_bodies_three_partitions(device)
        dt = 0.01

        # Mark body 0 static.
        am_h = bodies.access_mode.numpy().copy()
        am_h[0] = int(ACCESS_MODE_STATIC)
        bodies.access_mode.assign(am_h)
        v_orig_b0 = bodies.velocity.numpy()[0].copy()

        launch_broadcast_rigid_to_copy_states(cs, bodies, particles, num_bodies, dt)
        # Scribble divergent velocity into body 0's slots; writeback
        # must still skip the static body.
        vel_h = cs.velocity.numpy().copy()
        vel_h[0] = np.array([99.0, 99.0, 99.0], dtype=np.float32)
        vel_h[1] = np.array([99.0, 99.0, 99.0], dtype=np.float32)
        cs.velocity.assign(vel_h)

        launch_copy_state_into_rigids(cs, bodies, particles, num_bodies, 1.0 / dt)
        wp.synchronize_device(device)

        np.testing.assert_array_equal(bodies.velocity.numpy()[0], v_orig_b0)


if __name__ == "__main__":
    unittest.main()
