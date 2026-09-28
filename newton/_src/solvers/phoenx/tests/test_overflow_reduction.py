# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservation checks for direct writeback of rigid overflow copies."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_STATIC, ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.body import BodyContainer, body_container_zeros
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer, copy_state_container_zeros
from newton._src.solvers.phoenx.mass_splitting.kernels import (
    initialize_rigid_overflow_copy,
    launch_average_rigid_velocity_into_bodies,
    launch_broadcast_rigid_to_copy_states,
)
from newton._src.solvers.phoenx.particle import particle_container_zeros


@wp.kernel
def _initialize_contact_copies(
    copies: CopyStateContainer,
    bodies: BodyContainer,
    body_ids: wp.array[wp.int32],
    partition_ids: wp.array[wp.int32],
    dt: wp.float32,
):
    tid = wp.tid()
    initialize_rigid_overflow_copy(copies, bodies, body_ids[tid], partition_ids[tid], dt)


@unittest.skipUnless(wp.is_cuda_available(), "Rigid overflow reduction requires CUDA")
class TestOverflowReduction(unittest.TestCase):
    def test_contact_local_initialization_matches_broadcast(self) -> None:
        device = wp.get_cuda_device()
        bodies = body_container_zeros(3, device=device)
        bodies.position.assign(
            wp.array(np.asarray(((1, 2, 3), (4, 5, 6), (7, 8, 9)), dtype=np.float32), dtype=wp.vec3f, device=device)
        )
        bodies.orientation.assign(
            wp.array(np.asarray(((0, 0, 0, 1),) * 3, dtype=np.float32), dtype=wp.quatf, device=device)
        )
        bodies.velocity.assign(
            wp.array(np.asarray(((1, 0, 0), (0, 2, 0), (0, 0, 3)), dtype=np.float32), dtype=wp.vec3f, device=device)
        )
        bodies.angular_velocity.assign(
            wp.array(np.asarray(((0, 1, 0), (0, 0, 2), (3, 0, 0)), dtype=np.float32), dtype=wp.vec3f, device=device)
        )
        bodies.access_mode.assign(
            np.asarray((ACCESS_MODE_VELOCITY_LEVEL, ACCESS_MODE_VELOCITY_LEVEL, ACCESS_MODE_STATIC), dtype=np.int32)
        )

        broadcast = copy_state_container_zeros(capacity=4, num_nodes=3, device=device)
        contact_local = copy_state_container_zeros(capacity=4, num_nodes=3, device=device)
        for copies in (broadcast, contact_local):
            copies.count_per_node.assign(np.asarray((1, 2, 1), dtype=np.int32))
            copies.section_end.assign(np.asarray((1, 3, 4), dtype=np.int32))
            copies.slot_for_pid0.assign(np.asarray((0, 1, 3), dtype=np.int32))
            copies.partition_list.assign(np.asarray((0, 0, 1, 0), dtype=np.int32))
            copies.highest_index_in_use.assign(np.asarray((4,), dtype=np.int32))

        dt = 1.0 / 120.0
        launch_broadcast_rigid_to_copy_states(
            broadcast,
            bodies,
            particle_container_zeros(1, device=device),
            num_bodies=3,
            dt=dt,
        )
        wp.launch(
            _initialize_contact_copies,
            dim=4,
            inputs=[
                contact_local,
                bodies,
                wp.array(np.asarray((0, 1, 1, 2), dtype=np.int32), device=device),
                wp.array(np.asarray((0, 0, 1, 0), dtype=np.int32), device=device),
                wp.float32(dt),
            ],
            device=device,
        )
        for name in ("position", "orientation", "velocity", "angular_velocity", "access_mode"):
            np.testing.assert_array_equal(getattr(contact_local, name).numpy(), getattr(broadcast, name).numpy())

    def test_mass_weighted_momentum_and_static_body(self) -> None:
        device = wp.get_cuda_device()
        bodies = body_container_zeros(4, device=device)
        copies = copy_state_container_zeros(capacity=4, num_nodes=4, device=device)

        # Body 0 has no overflow contacts. Body 1 has two copies of equal
        # effective mass; body 2 has one; body 3 is static.
        copies.count_per_node.assign(np.asarray((0, 2, 1, 1), dtype=np.int32))
        copies.section_end.assign(np.asarray((0, 2, 3, 4), dtype=np.int32))
        copies.access_mode.assign(
            np.asarray(
                (ACCESS_MODE_VELOCITY_LEVEL,) * 3 + (ACCESS_MODE_STATIC,),
                dtype=np.int32,
            )
        )
        copy_velocity = np.asarray(((2, 0, 0), (4, 0, 0), (0, 6, 0), (99, 0, 0)), dtype=np.float32)
        copy_angular = np.asarray(((0, 2, 0), (0, 4, 0), (0, 0, 6), (99, 0, 0)), dtype=np.float32)
        copies.velocity.assign(wp.array(copy_velocity, dtype=wp.vec3f, device=device))
        copies.angular_velocity.assign(wp.array(copy_angular, dtype=wp.vec3f, device=device))
        bodies.velocity.assign(
            wp.array(
                np.asarray(((0, 0, 0), (0, 0, 0), (0, 0, 0), (7, 0, 0)), dtype=np.float32),
                dtype=wp.vec3f,
                device=device,
            )
        )

        launch_average_rigid_velocity_into_bodies(copies, bodies)
        velocity = bodies.velocity.numpy()
        angular = bodies.angular_velocity.numpy()
        np.testing.assert_array_equal(velocity[1], np.asarray((3, 0, 0), dtype=np.float32))
        np.testing.assert_array_equal(velocity[2], np.asarray((0, 6, 0), dtype=np.float32))
        np.testing.assert_array_equal(velocity[3], np.asarray((7, 0, 0), dtype=np.float32))
        np.testing.assert_array_equal(angular[1], np.asarray((0, 3, 0), dtype=np.float32))
        np.testing.assert_array_equal(angular[2], np.asarray((0, 0, 6), dtype=np.float32))
        np.testing.assert_array_equal(copies.velocity.numpy(), copy_velocity)

        # Each split copy carries half of body 1's mass (2 kg), so its
        # total momentum must equal the reconciled 2 kg body momentum.
        split_momentum = np.float32(1.0) * copy_velocity[0] + np.float32(1.0) * copy_velocity[1]
        np.testing.assert_array_equal(np.float32(2.0) * velocity[1], split_momentum)
