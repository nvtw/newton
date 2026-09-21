# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check external impulse timing and temporal substep pose snapshots."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, MOTION_STATIC, body_container_zeros
from newton._src.solvers.phoenx.simulation import PhoenXWorld


class TestTemporalForceTiming(unittest.TestCase):
    def test_force_impulse_and_snapshots(self):
        """Apply one outer impulse while retaining pose snapshots on every substep."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            for temporal in (False, True):
                with self.subTest(device=device, temporal=temporal):
                    self._check_case(device, temporal)

    def _check_case(self, device, temporal):
        bodies = body_container_zeros(2, device)
        bodies.motion_type.assign([MOTION_DYNAMIC, MOTION_STATIC])
        bodies.inverse_mass.assign([0.5, 0.0])
        bodies.inverse_inertia_world.assign([[1 / 3, 1 / 3, 1 / 3, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        bodies.force.assign([[2, 0, 0], [0, 0, 0]])
        bodies.torque.assign([[0, 6, 0], [0, 0, 0]])
        bodies.orientation.assign([[0, 0, 0, 1], [0, 0, 0, 1]])
        world = SimpleNamespace(
            bodies=bodies,
            num_bodies=2,
            num_particles=0,
            particles=None,
            _has_maximal_dynamic_bodies=True,
            _temporal_force_step=temporal,
            step_dt=0.03,
            substep_dt=0.01,
            device=device,
            gravity=wp.array([[0, 0, -10]], dtype=wp.vec3f, device=device),
        )
        for outer in range(2):
            for substep in range(3):
                world._current_substep_index = substep
                positions = np.array([[outer, substep, 1], [outer, substep, 0]], dtype=np.float32)
                bodies.position.assign(positions)
                PhoenXWorld._integrate_forces_and_gravity(world)
                elapsed = (outer + 1) * 0.03 if temporal else outer * 0.03 + (substep + 1) * 0.01
                velocity = bodies.velocity.numpy()
                spin = bodies.angular_velocity.numpy()
                # m=2; inertia=3. Account for actual external linear/angular impulse.
                np.testing.assert_allclose(2 * velocity[0], np.array([2, 0, -20]) * elapsed, atol=2e-7)
                np.testing.assert_allclose(3 * spin[0], np.array([0, 6, 0]) * elapsed, atol=2e-7)
                np.testing.assert_array_equal(velocity[1], 0)
                np.testing.assert_array_equal(spin[1], 0)
                np.testing.assert_array_equal(bodies.position_prev_substep.numpy(), positions)
                np.testing.assert_array_equal(bodies.orientation_prev_substep.numpy(), bodies.orientation.numpy())


if __name__ == "__main__":
    unittest.main()
