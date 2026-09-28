# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Kamino's gyroscopic angular-velocity update."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.math import compute_body_twist_update_with_eom


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


if __name__ == "__main__":
    unittest.main()
