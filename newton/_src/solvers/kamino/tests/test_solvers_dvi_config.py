# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused Kamino DVI Config tests."""

from newton._src.solvers.kamino.tests.dvi_test_helpers import *  # noqa: F403


class TestDVIConfig(DVITestCase):
    def test_00_config_selection(self):
        default_config = SolverKamino.Config(dynamics_solver="dvi")
        self.assertTrue(default_config.sparse_dynamics)
        self.assertTrue(default_config.sparse_jacobian)
        self.assertEqual(default_config.integrator, "moreau")
        self.assertEqual(default_config.dynamics.linear_solver_type, "CR")
        self.assertEqual(default_config.dynamics.linear_solver_kwargs, {"maxiter": 9})
        self.assertEqual(default_config.dvi.omega, 0.3)
        self.assertEqual(default_config.dvi.block_iterations, 16)
        self.assertEqual(default_config.dvi.contact_iterations, 2)
        self.assertEqual(default_config.dvi.bilateral_solve_period, 2)
        self.assertEqual(default_config.dvi.contact_jacobi_omega, 0.45)
        self.assertEqual(default_config.dvi.contact_jacobi_relaxation, 0.9)

        dense_config = SolverKamino.Config(
            dynamics_solver="dvi",
            sparse_dynamics=False,
            sparse_jacobian=False,
        )
        self.assertFalse(dense_config.sparse_dynamics)
        self.assertFalse(dense_config.sparse_jacobian)
        self.assertEqual(dense_config.integrator, "moreau")
        self.assertEqual(dense_config.dynamics.linear_solver_type, "LLTB")
        self.assertEqual(dense_config.dvi.block_iterations, 32)

        padmm_config = SolverKamino.Config()
        self.assertFalse(padmm_config.sparse_dynamics)
        self.assertFalse(padmm_config.sparse_jacobian)
        self.assertEqual(padmm_config.integrator, "euler")

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(max_iterations=32, tolerance=1e-4),
        )
        self.assertEqual(config.dynamics_solver, "dvi")
        self.assertEqual(config.dvi.max_iterations, 32)
        self.assertEqual(config.dvi.block_iterations, 32)
        self.assertEqual(config.dvi.contact_iterations, 4)
        self.assertEqual(config.dvi.bilateral_solve_period, 1)
        self.assertEqual(config.dvi.contact_jacobi_omega, 0.3)
        self.assertEqual(config.dvi.contact_jacobi_relaxation, 0.9)
        self.assertFalse(config.dvi.contact_block_preconditioner)
        self.assertEqual(config.dvi.contact_warmstart_method, "key_and_position_with_net_force_backup")
        self.assertFalse(config.dynamics.preconditioning)
        self.assertEqual(config.constraints.contact_recovery_speed, -1.0)
        self.assertEqual(config.constraints.contact_deep_recovery_gamma, -1.0)
        self.assertEqual(config.constraints.contact_deep_recovery_threshold, 0.0)

        sparse_config = SolverKamino.Config(dynamics_solver="dvi", sparse_dynamics=True, sparse_jacobian=True)
        self.assertTrue(sparse_config.sparse_dynamics)
        self.assertTrue(sparse_config.sparse_jacobian)
        with self.assertRaises(ValueError):
            SolverKamino.Config(
                dynamics_solver="dvi",
                dynamics=kamino_config.ConstrainedDynamicsConfig(preconditioning=True),
            )
        with self.assertRaises(ValueError):
            kamino_config.ConstraintStabilizationConfig(contact_deep_recovery_gamma=1.1)
        with self.assertRaises(ValueError):
            kamino_config.ConstraintStabilizationConfig(contact_deep_recovery_threshold=-1.0)
        invalid_dvi_configs = (
            {"max_iterations": 0},
            {"tolerance": -1.0},
            {"regularization": 0.0},
            {"omega": 0.0},
            {"omega": 2.1},
            {"block_iterations": 0},
            {"contact_iterations": 0},
            {"bilateral_solve_period": 0},
            {"contact_jacobi_omega": 0.0},
            {"contact_jacobi_omega": 2.1},
            {"contact_jacobi_relaxation": 0.0},
            {"contact_jacobi_relaxation": 1.1},
            {"warmstart_mode": "invalid"},
        )
        for kwargs in invalid_dvi_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                kamino_config.DVISolverConfig(**kwargs)
        for method in (
            "key_and_position",
            "geom_pair_net_force",
            "key_and_position_with_net_force_backup",
        ):
            self.assertEqual(
                kamino_config.DVISolverConfig(contact_warmstart_method=method).contact_warmstart_method, method
            )
        for method in ("reaction", "geom_pair_net_wrench"):
            with self.assertRaises(ValueError):
                kamino_config.DVISolverConfig(contact_warmstart_method=method)

        from types import SimpleNamespace  # noqa: PLC0415

        model_with_attrs = SimpleNamespace(
            kamino=SimpleNamespace(max_solver_iterations=wp.array([37], dtype=wp.int32, device=self.device))
        )
        self.assertEqual(kamino_config.DVISolverConfig.from_model(model_with_attrs).max_iterations, 37)


    def test_00a_multiworld_status_reduction_requires_all_worlds_converged(self):
        status = np.array(
            [(True, 2, 1.0e-5), (False, 7, 2.0e-3)],
            dtype=[("converged", np.bool_), ("iterations", np.int32), ("r_d", np.float32)],
        )

        reduced = _reduce_solver_status(status)

        self.assertFalse(reduced["converged"])
        self.assertEqual(reduced["iterations"], 7)
        self.assertAlmostEqual(reduced["r_d"], 2.0e-3)
