# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Routing tests for the slot-aware read/write helpers.

These exercise :func:`get_state_index` and the
``read_*_unified`` / ``write_*_unified`` helpers via tiny probe
kernels. The contracts under test:

* Disabled fast path (``highest_index_in_use[0] == 0``): all reads
  return body / particle direct storage, ``inv_factor == 1``,
  ``slot == -1``.
* Built graph: reads for known ``(node, parallel_id)`` pairs return
  the matching slot value; ``inv_factor == count`` (the node's slot
  count); ``slot >= 0``.
* Unknown ``parallel_id`` for a node with slots: falls through to
  direct, ``inv_factor == 1``.
* Particle branch: reads for ``node_id >= num_bodies`` index into
  ``particles`` at ``node_id - num_bodies``.
* Round-trip: ``write_*`` followed by ``read_*`` round-trips through
  either the slot or direct path as dictated by ``slot``.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

# -----------------------------------------------------------------------------
# Probe kernels: thread 0 issues a single read, writes the (value, inv_factor,
# slot) triple into output arrays. Using thread 0 keeps the test single-shot
# and removes any indexing race with parallel reads.
# -----------------------------------------------------------------------------
from newton._src.solvers.phoenx.body import (
    BodyContainer,
    body_container_zeros,
)
from newton._src.solvers.phoenx.mass_splitting.access import (
    read_angular_velocity_unified,
    read_orientation_unified,
    read_position_unified,
    read_velocity_unified,
    write_angular_velocity_unified,
    write_orientation_unified,
    write_position_unified,
    write_velocity_unified,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import (
    CopyStateContainer,
    copy_state_container_zeros,
)
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    build_interaction_graph,
    interaction_graph_scratch_zeros,
)
from newton._src.solvers.phoenx.mass_splitting.tests.test_interaction_graph import (
    _seed_pairs_direct,
)
from newton._src.solvers.phoenx.particle import (
    ParticleContainer,
    particle_container_zeros,
)


@wp.kernel(enable_backward=False)
def _probe_read_velocity_kernel(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
    out_value: wp.array[wp.vec3f],
    out_inv_factor: wp.array[wp.int32],
    out_slot: wp.array[wp.int32],
):
    if wp.tid() != 0:
        return
    v, f, s = read_velocity_unified(bodies, particles, copy_state, node_id, parallel_id, num_bodies)
    out_value[0] = v
    out_inv_factor[0] = f
    out_slot[0] = s


@wp.kernel(enable_backward=False)
def _probe_read_position_kernel(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
    out_value: wp.array[wp.vec3f],
    out_inv_factor: wp.array[wp.int32],
    out_slot: wp.array[wp.int32],
):
    if wp.tid() != 0:
        return
    v, f, s = read_position_unified(bodies, particles, copy_state, node_id, parallel_id, num_bodies)
    out_value[0] = v
    out_inv_factor[0] = f
    out_slot[0] = s


@wp.kernel(enable_backward=False)
def _probe_read_angular_velocity_kernel(
    bodies: BodyContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
    out_value: wp.array[wp.vec3f],
    out_inv_factor: wp.array[wp.int32],
    out_slot: wp.array[wp.int32],
):
    if wp.tid() != 0:
        return
    w, f, s = read_angular_velocity_unified(bodies, copy_state, node_id, parallel_id, num_bodies)
    out_value[0] = w
    out_inv_factor[0] = f
    out_slot[0] = s


@wp.kernel(enable_backward=False)
def _probe_read_orientation_kernel(
    bodies: BodyContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
    out_value: wp.array[wp.quatf],
    out_inv_factor: wp.array[wp.int32],
    out_slot: wp.array[wp.int32],
):
    if wp.tid() != 0:
        return
    q, f, s = read_orientation_unified(bodies, copy_state, node_id, parallel_id, num_bodies)
    out_value[0] = q
    out_inv_factor[0] = f
    out_slot[0] = s


@wp.kernel(enable_backward=False)
def _probe_roundtrip_velocity_kernel(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
    write_value: wp.vec3f,
    out_value: wp.array[wp.vec3f],
):
    if wp.tid() != 0:
        return
    # Warp's codegen rejects ``_`` as a throwaway (treats every name as
    # a single typed symbol). Use distinct names for the unused tuple
    # slots.
    _v0, _f0, slot = read_velocity_unified(bodies, particles, copy_state, node_id, parallel_id, num_bodies)
    write_velocity_unified(bodies, particles, copy_state, node_id, slot, num_bodies, write_value)
    v1, _f1, _s1 = read_velocity_unified(bodies, particles, copy_state, node_id, parallel_id, num_bodies)
    out_value[0] = v1


@wp.kernel(enable_backward=False)
def _probe_roundtrip_position_kernel(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
    write_value: wp.vec3f,
    out_value: wp.array[wp.vec3f],
):
    if wp.tid() != 0:
        return
    _v0, _f0, slot = read_position_unified(bodies, particles, copy_state, node_id, parallel_id, num_bodies)
    write_position_unified(bodies, particles, copy_state, node_id, slot, num_bodies, write_value)
    v1, _f1, _s1 = read_position_unified(bodies, particles, copy_state, node_id, parallel_id, num_bodies)
    out_value[0] = v1


@wp.kernel(enable_backward=False)
def _probe_roundtrip_angular_velocity_kernel(
    bodies: BodyContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
    write_value: wp.vec3f,
    out_value: wp.array[wp.vec3f],
):
    if wp.tid() != 0:
        return
    _w0, _f0, slot = read_angular_velocity_unified(bodies, copy_state, node_id, parallel_id, num_bodies)
    write_angular_velocity_unified(bodies, copy_state, node_id, slot, write_value)
    w1, _f1, _s1 = read_angular_velocity_unified(bodies, copy_state, node_id, parallel_id, num_bodies)
    out_value[0] = w1


@wp.kernel(enable_backward=False)
def _probe_roundtrip_orientation_kernel(
    bodies: BodyContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
    write_value: wp.quatf,
    out_value: wp.array[wp.quatf],
):
    if wp.tid() != 0:
        return
    _q0, _f0, slot = read_orientation_unified(bodies, copy_state, node_id, parallel_id, num_bodies)
    write_orientation_unified(bodies, copy_state, node_id, slot, write_value)
    q1, _f1, _s1 = read_orientation_unified(bodies, copy_state, node_id, parallel_id, num_bodies)
    out_value[0] = q1


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _setup_fixture(device, build_graph: bool = True):
    """2 bodies + 1 particle. Body 0 has parallel keys {0, 5}; body 1 has
    {0, 2, 7}; particle (unified node 2) has no slots.

    Returns (bodies, particles, copy_state, num_bodies).
    """
    num_bodies = 2
    num_particles = 1
    num_nodes = num_bodies + num_particles
    capacity = 16

    bodies = body_container_zeros(num_bodies=num_bodies, device=device)
    particles = particle_container_zeros(num_particles=num_particles, device=device)
    copy_state = copy_state_container_zeros(capacity=capacity, num_nodes=num_nodes, device=device)

    bodies.position.assign(np.asarray([(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)], dtype=np.float32))
    bodies.orientation.assign(np.asarray([(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)], dtype=np.float32))
    bodies.velocity.assign(np.asarray([(0.5, 0.0, 0.0), (0.0, 0.7, 0.0)], dtype=np.float32))
    bodies.angular_velocity.assign(np.asarray([(0.1, 0.2, 0.3), (0.0, 0.0, 0.5)], dtype=np.float32))

    particles.position.assign(np.asarray([(5.0, 5.0, 5.0)], dtype=np.float32))
    particles.velocity.assign(np.asarray([(1.0, 0.0, -1.0)], dtype=np.float32))

    if build_graph:
        scratch = interaction_graph_scratch_zeros(capacity=capacity, device=device)
        # Body 0 in partitions 0, 5; body 1 in 0, 2, 7; particle (node 2) none.
        pairs = [(0, 0), (0, 5), (1, 0), (1, 2), (1, 7)]
        _seed_pairs_direct(scratch, pairs, device)
        build_interaction_graph(scratch, copy_state)
        wp.synchronize_device(device)
        # Hand-write distinct values into each slot so the read tests can
        # detect mis-routing. slot k gets velocity (10+k, 0, 0).
        n_slots = int(copy_state.highest_index_in_use.numpy()[0])
        vel_h = copy_state.velocity.numpy().copy()
        for k in range(n_slots):
            vel_h[k] = np.array([10.0 + float(k), 0.0, 0.0], dtype=np.float32)
        copy_state.velocity.assign(vel_h)
        pos_h = copy_state.position.numpy().copy()
        for k in range(n_slots):
            pos_h[k] = np.array([20.0 + float(k), 0.0, 0.0], dtype=np.float32)
        copy_state.position.assign(pos_h)
        ang_h = copy_state.angular_velocity.numpy().copy()
        for k in range(n_slots):
            ang_h[k] = np.array([30.0 + float(k), 0.0, 0.0], dtype=np.float32)
        copy_state.angular_velocity.assign(ang_h)
        orient_h = copy_state.orientation.numpy().copy()
        for k in range(n_slots):
            orient_h[k] = np.array([0.0, 0.0, 0.0, float(40 + k)], dtype=np.float32)
        copy_state.orientation.assign(orient_h)

    return bodies, particles, copy_state, num_bodies


def _read_probe_velocity(bodies, particles, copy_state, node_id, parallel_id, num_bodies, device):
    out_v = wp.zeros(1, dtype=wp.vec3f, device=device)
    out_f = wp.zeros(1, dtype=wp.int32, device=device)
    out_s = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _probe_read_velocity_kernel,
        dim=1,
        inputs=[
            bodies,
            particles,
            copy_state,
            wp.int32(node_id),
            wp.int32(parallel_id),
            wp.int32(num_bodies),
            out_v,
            out_f,
            out_s,
        ],
        device=device,
    )
    return out_v.numpy()[0], int(out_f.numpy()[0]), int(out_s.numpy()[0])


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX mass-splitting tests are CUDA-only.",
)
class TestAccessRouting(unittest.TestCase):
    def test_disabled_fast_path_returns_direct(self):
        # No build → highest_index_in_use == 0. Every read returns direct
        # storage with inv_factor=1, slot=-1.
        device = wp.get_preferred_device()
        bodies, particles, copy_state, num_bodies = _setup_fixture(device, build_graph=False)
        v, f, s = _read_probe_velocity(
            bodies, particles, copy_state, node_id=0, parallel_id=0, num_bodies=num_bodies, device=device
        )
        np.testing.assert_allclose(v, [0.5, 0.0, 0.0])
        self.assertEqual(f, 1)
        self.assertEqual(s, -1)

    def test_disabled_fast_path_particle_branch(self):
        device = wp.get_preferred_device()
        bodies, particles, copy_state, num_bodies = _setup_fixture(device, build_graph=False)
        v, f, s = _read_probe_velocity(
            bodies, particles, copy_state, node_id=2, parallel_id=0, num_bodies=num_bodies, device=device
        )
        # Particle 0 velocity is (1, 0, -1).
        np.testing.assert_allclose(v, [1.0, 0.0, -1.0])
        self.assertEqual(f, 1)
        self.assertEqual(s, -1)

    def test_built_graph_hits_slot(self):
        # Body 1 in partitions {0, 2, 7}. Sections: [start, end) = [2, 5).
        # partition_list[2:5] = [0, 2, 7]. Slot for (body=1, parallel=2)
        # is slot 3. Slot 3 was hand-stamped velocity (13, 0, 0).
        device = wp.get_preferred_device()
        bodies, particles, copy_state, num_bodies = _setup_fixture(device, build_graph=True)
        v, f, s = _read_probe_velocity(
            bodies, particles, copy_state, node_id=1, parallel_id=2, num_bodies=num_bodies, device=device
        )
        self.assertEqual(s, 3, msg="expected slot 3 for (body=1, parallel=2)")
        # Body 1 has 3 partition slots so inv_factor=3.
        self.assertEqual(f, 3)
        np.testing.assert_allclose(v, [13.0, 0.0, 0.0])

    def test_built_graph_body_0_slot(self):
        # Body 0 in {0, 5}. partition_list[0:2] = [0, 5]. (body=0, parallel=0)
        # → slot 0. inv_factor = 2.
        device = wp.get_preferred_device()
        bodies, particles, copy_state, num_bodies = _setup_fixture(device, build_graph=True)
        v, f, s = _read_probe_velocity(
            bodies, particles, copy_state, node_id=0, parallel_id=0, num_bodies=num_bodies, device=device
        )
        self.assertEqual(s, 0)
        self.assertEqual(f, 2)
        np.testing.assert_allclose(v, [10.0, 0.0, 0.0])

    def test_unknown_parallel_id_falls_through(self):
        # Body 1 has partitions {0, 2, 7}. parallel=99 not present.
        # Read must fall back to bodies.velocity[1].
        device = wp.get_preferred_device()
        bodies, particles, copy_state, num_bodies = _setup_fixture(device, build_graph=True)
        v, f, s = _read_probe_velocity(
            bodies, particles, copy_state, node_id=1, parallel_id=99, num_bodies=num_bodies, device=device
        )
        self.assertEqual(s, -1)
        self.assertEqual(f, 1)
        np.testing.assert_allclose(v, [0.0, 0.7, 0.0])

    def test_particle_node_with_no_slots_falls_through(self):
        # Particle node (unified id 2) has no entries in the graph; read
        # falls through to particles.velocity[0].
        device = wp.get_preferred_device()
        bodies, particles, copy_state, num_bodies = _setup_fixture(device, build_graph=True)
        v, f, s = _read_probe_velocity(
            bodies, particles, copy_state, node_id=2, parallel_id=0, num_bodies=num_bodies, device=device
        )
        self.assertEqual(s, -1)
        self.assertEqual(f, 1)
        np.testing.assert_allclose(v, [1.0, 0.0, -1.0])

    def test_position_read_hits_slot(self):
        device = wp.get_preferred_device()
        bodies, particles, copy_state, num_bodies = _setup_fixture(device, build_graph=True)
        out_v = wp.zeros(1, dtype=wp.vec3f, device=device)
        out_f = wp.zeros(1, dtype=wp.int32, device=device)
        out_s = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            _probe_read_position_kernel,
            dim=1,
            inputs=[
                bodies,
                particles,
                copy_state,
                wp.int32(0),
                wp.int32(5),  # body 0, parallel 5 → slot 1
                wp.int32(num_bodies),
                out_v,
                out_f,
                out_s,
            ],
            device=device,
        )
        self.assertEqual(int(out_s.numpy()[0]), 1)
        np.testing.assert_allclose(out_v.numpy()[0], [21.0, 0.0, 0.0])

    def test_angular_velocity_read_hits_slot(self):
        device = wp.get_preferred_device()
        bodies, _particles, copy_state, num_bodies = _setup_fixture(device, build_graph=True)
        out_w = wp.zeros(1, dtype=wp.vec3f, device=device)
        out_f = wp.zeros(1, dtype=wp.int32, device=device)
        out_s = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            _probe_read_angular_velocity_kernel,
            dim=1,
            inputs=[
                bodies,
                copy_state,
                wp.int32(1),
                wp.int32(7),  # body 1, parallel 7 → slot 4
                wp.int32(num_bodies),
                out_w,
                out_f,
                out_s,
            ],
            device=device,
        )
        self.assertEqual(int(out_s.numpy()[0]), 4)
        np.testing.assert_allclose(out_w.numpy()[0], [34.0, 0.0, 0.0])

    def test_orientation_read_hits_slot(self):
        device = wp.get_preferred_device()
        bodies, _particles, copy_state, num_bodies = _setup_fixture(device, build_graph=True)
        out_q = wp.zeros(1, dtype=wp.quatf, device=device)
        out_f = wp.zeros(1, dtype=wp.int32, device=device)
        out_s = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            _probe_read_orientation_kernel,
            dim=1,
            inputs=[
                bodies,
                copy_state,
                wp.int32(0),
                wp.int32(0),  # body 0, parallel 0 → slot 0
                wp.int32(num_bodies),
                out_q,
                out_f,
                out_s,
            ],
            device=device,
        )
        self.assertEqual(int(out_s.numpy()[0]), 0)
        np.testing.assert_allclose(out_q.numpy()[0], [0.0, 0.0, 0.0, 40.0])

    def test_writeback_routes_to_slot_when_active(self):
        device = wp.get_preferred_device()
        bodies, particles, copy_state, num_bodies = _setup_fixture(device, build_graph=True)
        out_v = wp.zeros(1, dtype=wp.vec3f, device=device)
        wp.launch(
            _probe_roundtrip_velocity_kernel,
            dim=1,
            inputs=[
                bodies,
                particles,
                copy_state,
                wp.int32(1),
                wp.int32(7),  # body 1, parallel 7 → slot 4
                wp.int32(num_bodies),
                wp.vec3f(99.0, 88.0, 77.0),
                out_v,
            ],
            device=device,
        )
        np.testing.assert_allclose(out_v.numpy()[0], [99.0, 88.0, 77.0])
        # Body 1's direct velocity should be unchanged (writeback went to slot).
        np.testing.assert_allclose(bodies.velocity.numpy()[1], [0.0, 0.7, 0.0])
        # Slot 4 should hold the new value.
        np.testing.assert_allclose(copy_state.velocity.numpy()[4], [99.0, 88.0, 77.0])

    def test_writeback_routes_to_body_when_no_slot(self):
        # parallel=99 not in body 1's slots → write goes to bodies.velocity[1].
        device = wp.get_preferred_device()
        bodies, particles, copy_state, num_bodies = _setup_fixture(device, build_graph=True)
        out_v = wp.zeros(1, dtype=wp.vec3f, device=device)
        wp.launch(
            _probe_roundtrip_velocity_kernel,
            dim=1,
            inputs=[
                bodies,
                particles,
                copy_state,
                wp.int32(1),
                wp.int32(99),
                wp.int32(num_bodies),
                wp.vec3f(55.0, 66.0, 77.0),
                out_v,
            ],
            device=device,
        )
        np.testing.assert_allclose(bodies.velocity.numpy()[1], [55.0, 66.0, 77.0])
        np.testing.assert_allclose(out_v.numpy()[0], [55.0, 66.0, 77.0])

    def test_writeback_routes_to_particle_when_disabled(self):
        device = wp.get_preferred_device()
        bodies, particles, copy_state, num_bodies = _setup_fixture(device, build_graph=False)
        out_v = wp.zeros(1, dtype=wp.vec3f, device=device)
        wp.launch(
            _probe_roundtrip_velocity_kernel,
            dim=1,
            inputs=[
                bodies,
                particles,
                copy_state,
                wp.int32(2),  # particle node
                wp.int32(0),
                wp.int32(num_bodies),
                wp.vec3f(7.7, 8.8, 9.9),
                out_v,
            ],
            device=device,
        )
        np.testing.assert_allclose(particles.velocity.numpy()[0], [7.7, 8.8, 9.9])

    def test_graph_capture_safe(self):
        # Same probe inside a captured graph — must succeed and round-trip
        # identically (the load-bearing capture-safety assertion).
        device = wp.get_preferred_device()
        bodies, particles, copy_state, num_bodies = _setup_fixture(device, build_graph=True)
        out_v = wp.zeros(1, dtype=wp.vec3f, device=device)

        # Warm-up to JIT.
        wp.launch(
            _probe_read_velocity_kernel,
            dim=1,
            inputs=[
                bodies,
                particles,
                copy_state,
                wp.int32(0),
                wp.int32(5),
                wp.int32(num_bodies),
                out_v,
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
            ],
            device=device,
        )
        out_v.zero_()

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                _probe_read_velocity_kernel,
                dim=1,
                inputs=[
                    bodies,
                    particles,
                    copy_state,
                    wp.int32(0),
                    wp.int32(5),
                    wp.int32(num_bodies),
                    out_v,
                    wp.zeros(1, dtype=wp.int32, device=device),
                    wp.zeros(1, dtype=wp.int32, device=device),
                ],
                device=device,
            )
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        # Body 0, parallel 5 → slot 1, velocity = (11, 0, 0).
        np.testing.assert_allclose(out_v.numpy()[0], [11.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
