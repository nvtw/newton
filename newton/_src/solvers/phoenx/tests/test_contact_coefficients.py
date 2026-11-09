# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact spring coefficients and their sweep-schedule stability bound."""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact_solve import (
    _default_contact_solver_coefficients,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    contact_spring_omega_dt,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_soft_coefficients,
    contact_container_set_sweeps,
)


@wp.kernel
def _compare_contact_coefficients(
    cc: ContactContainer,
    idts: wp.array[wp.float32],
    hertz_per_rate: wp.float32,
    specialized: wp.array2d[wp.float32],
    reference: wp.array2d[wp.float32],
):
    i = wp.tid()
    idt = idts[i]
    mass_coeff, impulse_coeff = _default_contact_solver_coefficients(cc, idt)
    bias_rate, _mass_coeff, _impulse_coeff = cc_soft_coefficients(cc, idt)
    specialized[i, 0] = bias_rate / idt
    specialized[i, 1] = mass_coeff
    specialized[i, 2] = impulse_coeff
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
        hertz_per_rate * idt, DEFAULT_DAMPING_RATIO, 1.0 / idt
    )
    reference[i, 0] = bias_rate / idt
    reference[i, 1] = mass_coeff
    reference[i, 2] = impulse_coeff


def _substep_map(d: float, omega_dt: float, solver_iterations: int, velocity_iterations: int) -> np.ndarray:
    """Substep map of one Delassus mode (gap, gap rate, impulse) with ``dt = 1``."""
    a1 = 2.0 + omega_dt
    a2 = omega_dt * a1
    beta, mass, leak = omega_dt / a1, a2 / (1.0 + a2), 1.0 / (1.0 + a2)
    columns = []
    for c0, u0, lam0 in np.eye(3):
        lam = lam0
        u = u0 + d * lam  # warm start
        for _ in range(solver_iterations):
            dlam = -mass * (u + beta * c0) - leak * lam
            lam += dlam
            u += d * dlam
        c = c0 + u
        for _ in range(velocity_iterations):
            dlam = -u
            lam += dlam
            u += d * dlam
        columns.append((c, u, lam))
    return np.array(columns).T


def _worst_growth(omega_dt: float, solver_iterations: int, velocity_iterations: int) -> float:
    return max(
        float(np.abs(np.linalg.eigvals(_substep_map(d, omega_dt, solver_iterations, velocity_iterations))).max())
        for d in np.geomspace(1.0e-4, 1.0, 400)
    )


class TestContactSpringBound(unittest.TestCase):
    def test_bound_is_the_resting_stability_limit(self):
        """Below the bound every mode decays; just above it a slow mode grows."""
        for sweeps in ((4, 1), (2, 1), (1, 1), (4, 3), (6, 1)):
            omega_dt = contact_spring_omega_dt(*sweeps)
            with self.subTest(sweeps=sweeps):
                self.assertLess(omega_dt, math.pi)
                self.assertLessEqual(_worst_growth(0.97 * omega_dt, *sweeps), 1.0)
                self.assertGreater(_worst_growth(min(1.05 * omega_dt, math.pi), *sweeps), 1.0)

    def test_bound_follows_the_sweep_schedule(self):
        self.assertAlmostEqual(contact_spring_omega_dt(4, 1) / math.pi, 0.780, places=3)
        # One biased and one relax sweep: Box2D's quarter of the substep rate.
        self.assertAlmostEqual(contact_spring_omega_dt(1, 1) / (2.0 * math.pi), 0.236, places=3)
        self.assertEqual(contact_spring_omega_dt(7, 1), math.pi)
        self.assertEqual(contact_spring_omega_dt(10, 1), math.pi)
        self.assertLess(contact_spring_omega_dt(4, 0), contact_spring_omega_dt(4, 1))
        self.assertLess(contact_spring_omega_dt(4, 1), contact_spring_omega_dt(4, 4))


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX contact tests require CUDA")
class TestDefaultContactCoefficients(unittest.TestCase):
    def test_container_coefficients_match_reference_under_graph_capture(self):
        idts = wp.array(np.array([1.0, 60.0, 300.0, 480.0, 1.0e6], dtype=np.float32), device="cuda:0")
        for sweeps in ((4, 1), (10, 1)):
            with self.subTest(sweeps=sweeps):
                cc = ContactContainer()
                contact_container_set_sweeps(cc, *sweeps)
                hertz_per_rate = contact_spring_omega_dt(*sweeps) / (2.0 * math.pi)
                specialized = wp.empty((len(idts), 3), dtype=wp.float32, device="cuda:0")
                reference = wp.empty_like(specialized)
                inputs = [cc, idts, hertz_per_rate, specialized, reference]
                with wp.ScopedCapture(device="cuda:0") as capture:
                    wp.launch(_compare_contact_coefficients, dim=len(idts), inputs=inputs, device="cuda:0")
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(specialized.numpy(), reference.numpy(), rtol=2.0e-5, atol=0.0)


if __name__ == "__main__":
    unittest.main()
