# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Kamino core math's gyroscopic angular-velocity update."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.math import _solve_gyroscopic_midpoint, compute_body_twist_update_with_eom


@wp.kernel
def _update_free_body(omega: wp.array[wp.vec3f], result: wp.array[wp.vec3f]):
    inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0)
    inverse_inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5)
    spin = omega[0]
    _linear, next_omega = compute_body_twist_update_with_eom(
        1.0 / 60.0,
        wp.vec3f(0.0),
        1.0,
        inertia,
        inverse_inertia,
        wp.spatial_vectorf(0.0, 0.0, 0.0, spin[0], spin[1], spin[2]),
        wp.spatial_vectorf(0.0),
    )
    result[0] = next_omega


class TestKaminoGyroscopicMidpoint(unittest.TestCase):
    def test_backtracking_converges_for_large_gyroscopic_torque(self):
        """Converge on a midpoint root when undamped Newton iterations diverge."""
        initial = np.array([9.870089476365965, -166.99239553143286, 41.82976545391431])
        torque = np.array([-959.912607906128, -823.5785208912565, -426.52546511027697])
        inertia = np.diag([1.0, 2.0, 4.0])
        dt = 0.02
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            with self.subTest(device=device):
                result = wp.empty(1, dtype=wp.vec3f, device=device)
                converged = wp.empty(1, dtype=bool, device=device)
                wp.launch(_solve_with_backtracking, dim=1, outputs=[result, converged], device=device)
                self.assertTrue(converged.numpy()[0])
                updated = result.numpy()[0]
                midpoint = 0.5 * (initial + updated)
                residual = inertia @ (updated - initial) - dt * (torque - np.cross(midpoint, inertia @ midpoint))
                np.testing.assert_allclose(residual, 0.0, atol=2.0e-3)
                energy_change = 0.5 * (updated @ inertia @ updated - initial @ inertia @ initial)
                self.assertAlmostEqual(energy_change, dt * torque @ midpoint, delta=0.1)

    def test_high_spin_preserves_rotational_energy(self):
        """Preserve torque-free energy when midpoint fixed-point iteration diverges."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            with self.subTest(device=device):
                omega = wp.array([[1.0, 0.0, 240.0]], dtype=wp.vec3f, device=device)
                result = wp.empty(1, dtype=wp.vec3f, device=device)
                wp.launch(_update_free_body, dim=1, inputs=[omega, result], device=device)
                next_omega = result.numpy()[0]
                self.assertTrue(np.isfinite(next_omega).all())
                self.assertAlmostEqual(float(next_omega[0] ** 2 + next_omega[1] ** 2), 1.0, delta=1.0e-3)
                np.testing.assert_allclose(next_omega, [-0.6, 0.8, 240.0], rtol=1.0e-4, atol=1.0e-4)


@wp.kernel
def _solve_with_backtracking(result: wp.array[wp.vec3f], converged: wp.array[bool]):
    inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0)
    inverse_inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.25)
    updated, success = _solve_gyroscopic_midpoint(
        0.02,
        inertia,
        inverse_inertia,
        wp.vec3f(9.870089476365965, -166.99239553143286, 41.82976545391431),
        wp.vec3f(-959.912607906128, -823.5785208912565, -426.52546511027697),
    )
    result[0] = updated
    converged[0] = success


if __name__ == "__main__":
    unittest.main()
