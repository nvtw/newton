# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check analytical free joint projection across physical mass ratios."""

import unittest

import numpy as np
import warp as wp

import newton
from local_studies.colibri.bilateral_pgs import install_fused


class TestBilateralMassRatio(unittest.TestCase):
    def test_free_fixed_pair_projection(self):
        """Recover common velocity and preserve momentum without mass scaling."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        for root_mass in (1.0e-6, 1.0, 1.0e3, 1.0e6):
            with self.subTest(root_mass=root_mass):
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                root = builder.add_link(
                    mass=root_mass, inertia=wp.mat33(root_mass, 0.0, 0.0, 0.0, root_mass, 0.0, 0.0, 0.0, root_mass)
                )
                tip = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                builder.add_articulation([builder.add_joint_free(root), builder.add_joint_fixed(root, tip)])
                model = builder.finalize(device="cuda:0")
                solver = newton.solvers.SolverPhoenX(
                    model,
                    articulation_mode="maximal",
                    step_layout="single_world",
                    substeps=1,
                    solver_iterations=1,
                    sor_boost=1.0,
                    prepare_refresh_stride=1,
                )
                install_fused(solver)
                state = model.state()
                velocity = np.array([0.3, -0.2, 0.1, -0.1, 0.2, 0.3], dtype=np.float32)
                qd = state.body_qd.numpy()
                qd[:] = 0.0
                qd[tip] = velocity
                state.body_qd.assign(qd)
                state.clear_forces()
                solver.step(state, state, model.control(), None, 0.001)
                after = state.body_qd.numpy().astype(np.float64)
                expected = np.tile(velocity.astype(np.float64) / (root_mass + 1.0), (2, 1))
                np.testing.assert_allclose(after, expected, rtol=3.0e-5, atol=2.0e-7)
                np.testing.assert_allclose(root_mass * after[root] + after[tip], velocity, rtol=5.0e-6, atol=2.0e-7)
                energy_before = 0.5 * np.dot(velocity, velocity)
                energy_after = 0.5 * (root_mass * np.dot(after[root], after[root]) + np.dot(after[tip], after[tip]))
                self.assertLessEqual(energy_after, energy_before * (1.0 + 1.0e-6))


if __name__ == "__main__":
    unittest.main()
