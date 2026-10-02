# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for gyroscopic torque in Kamino's dual assembly."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.dynamics.dual import (
    _build_generalized_free_velocity,
    _build_nonlinear_generalized_force,
)


class TestDualGyroscopicMidpoint(unittest.TestCase):
    def test_free_velocity_uses_midpoint_predictor(self):
        """Evaluate gyroscopic torque using external and actuation midpoint acceleration."""
        inertia = np.diag([1.0, 2.0, 4.0])
        inverse_inertia = np.linalg.inv(inertia)
        velocity = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        external = np.array([2.0, 3.0, 4.0, 8.0, 12.0, 16.0])
        actuation = np.array([1.0, -1.0, 2.0, 4.0, -4.0, 8.0])
        gravity = np.array([0.0, 0.0, -9.81])
        dt = 0.02
        midpoint = velocity[3:] + 0.5 * dt * inverse_inertia @ (external[3:] + actuation[3:])
        wrench = external + actuation + np.concatenate([2.0 * gravity, -np.cross(midpoint, inertia @ midpoint)])
        expected = velocity + dt * np.concatenate([0.5 * wrench[:3], inverse_inertia @ wrench[3:]])
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            with self.subTest(device=device):
                inputs = [
                    wp.array([dt], dtype=float, device=device),
                    wp.array([gravity], dtype=wp.vec3f, device=device),
                    wp.array([0], dtype=int, device=device),
                    wp.array([2.0], dtype=float, device=device),
                    wp.array([0.5], dtype=float, device=device),
                    wp.array([velocity], dtype=wp.spatial_vectorf, device=device),
                    wp.array([inertia], dtype=wp.mat33f, device=device),
                    wp.array([inverse_inertia], dtype=wp.mat33f, device=device),
                    wp.array([external], dtype=wp.spatial_vectorf, device=device),
                    wp.array([actuation], dtype=wp.spatial_vectorf, device=device),
                ]
                result = wp.empty(1, dtype=wp.spatial_vectorf, device=device)
                wp.launch(_build_generalized_free_velocity, dim=1, inputs=inputs, outputs=[result], device=device)
                np.testing.assert_allclose(result.numpy()[0], expected, rtol=1.0e-6, atol=1.0e-6)
                impulse = wp.empty_like(result)
                wp.launch(
                    _build_nonlinear_generalized_force,
                    dim=1,
                    inputs=inputs[:4] + inputs[5:],
                    outputs=[impulse],
                    device=device,
                )
                np.testing.assert_allclose(impulse.numpy()[0], dt * wrench, rtol=1.0e-6, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
