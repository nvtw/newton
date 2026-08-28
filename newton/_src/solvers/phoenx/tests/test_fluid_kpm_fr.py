# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from numpy.polynomial.legendre import Legendre

from newton._src.solvers.phoenx.fluid import KPMFR2D, KPMFR3D, KPMFR2DConfig, KPMFR3DConfig
from newton._src.solvers.phoenx.fluid.operators import fr_operators


class TestKPMFR2D(unittest.TestCase):
    def test_operators_differentiate_polynomials(self):
        """Differentiate polynomials exactly on the solution points."""
        for order in (3, 4, 5, 6):
            points, derivative, _, _, _ = fr_operators(order)
            for degree in range(order):
                values = points**degree
                expected = np.zeros_like(points) if degree == 0 else degree * points ** (degree - 1)
                np.testing.assert_allclose(derivative @ values, expected, atol=2.0e-5)

    def test_uniform_flow_is_preserved(self):
        """Preserve a uniform periodic flow through one update."""
        solver = KPMFR2D(KPMFR2DConfig((2, 2), order=3), device="cpu")
        solver.initialize(1.0, (0.1, -0.02))
        initial = solver.state.numpy()
        solver.step()
        np.testing.assert_allclose(solver.state.numpy(), initial, atol=2.0e-6)

    def test_obstacle_penalizes_momentum(self):
        """Remove momentum at solid solution points."""
        solver = KPMFR2D(KPMFR2DConfig((2, 2), order=3), device="cpu")
        solver.initialize(1.0, (0.1, 0.02))
        mask = np.ones(solver.volume_fraction.shape, dtype=np.float16)
        solver.volume_fraction.assign(mask)
        solver.step()
        state = solver.state.numpy()
        np.testing.assert_allclose(state[0], 1.0, atol=2.0e-6)
        np.testing.assert_allclose(state[1:], 0.0, atol=2.0e-6)


class TestKPMFR3D(unittest.TestCase):
    def test_uniform_flow_is_preserved(self):
        """Preserve a uniform periodic flow through one 3D update."""
        solver = KPMFR3D(KPMFR3DConfig((2, 2, 2), order=3), device="cpu")
        solver.initialize(1.0, (0.1, -0.02, 0.03))
        initial = solver.state.numpy()
        solver.step()
        np.testing.assert_array_equal(solver.state.numpy(), initial)

    def test_obstacle_penalizes_momentum(self):
        """Remove all three momentum components at solid points."""
        solver = KPMFR3D(KPMFR3DConfig((2, 2, 2), order=3), device="cpu")
        solver.initialize(1.0, (0.1, -0.02, 0.03))
        solver.volume_fraction.fill_(1.0)
        solver.step()
        state = solver.state.numpy()
        np.testing.assert_allclose(state[0], 1.0, atol=2.0e-6)
        np.testing.assert_allclose(state[1:], 0.0, atol=2.0e-6)

    def test_nonuniform_update_is_deterministic_and_conservative(self):
        """Preserve weighted mass and repeat a nonuniform update exactly."""
        config = KPMFR3DConfig((2, 2, 2), order=3)
        first = KPMFR3D(config, device="cpu")
        second = KPMFR3D(config, device="cpu")
        rng = np.random.default_rng(7)
        state = np.empty(first.state.shape, dtype=np.float32)
        state[0] = 1.0 + 0.01 * rng.standard_normal(state.shape[1])
        for axis, mean in enumerate((0.08, 0.0, 0.0)):
            state[axis + 1] = state[0] * (mean + 0.005 * rng.standard_normal(state.shape[1]))
        first.state.assign(state)
        second.state.assign(state)
        polynomial = Legendre.basis(config.order - 1)(first.points)
        weights = 2.0 / (config.order * (config.order - 1) * polynomial**2)
        weights_3d = np.einsum("z,y,x->zyx", weights, weights, weights).ravel()
        mass_before = np.sum(state[0].reshape(-1, config.order**3) * weights_3d)
        first.step()
        second.step()
        result = first.state.numpy()
        np.testing.assert_array_equal(result, second.state.numpy())
        self.assertTrue(np.isfinite(result).all())
        mass_after = np.sum(result[0].reshape(-1, config.order**3) * weights_3d)
        self.assertAlmostEqual(mass_after, mass_before, places=5)


if __name__ == "__main__":
    unittest.main()
