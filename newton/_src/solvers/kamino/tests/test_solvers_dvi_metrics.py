# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused Kamino DVI Metrics tests."""

from newton._src.solvers.kamino.tests.dvi_test_helpers import *  # noqa: F403


class TestDVIMetrics(DVITestCase):
    def test_03h_dvi_canonical_contact_solution_metrics(self):
        for builder_fn, max_world_contacts in (
            (basics.build_box_on_plane, 4),
            (basics.build_boxes_hinged, 8),
        ):
            for sparse in (False, True):
                with self.subTest(builder=builder_fn.__name__, sparse=sparse):
                    test = TestSetup(
                        builder_fn=builder_fn,
                        max_world_contacts=max_world_contacts,
                        gravity=True,
                        perturb=True,
                        device=self.device,
                        sparse=sparse,
                    )
                    test.build()
                    config = SolverKamino.Config(
                        dynamics_solver="dvi",
                        sparse_dynamics=sparse,
                        sparse_jacobian=sparse,
                    ).dvi
                    solver = _solve_dvi(test.model, test.problem, config=config)
                    solution_metrics = _evaluate_solution_metrics(test, solver)

                    _assert_solution_finite(self, solver)
                    for name, value in solution_metrics.items():
                        self.assertTrue(np.isfinite(value), msg=f"{name}={value}")

                    # DVI trades some contact accuracy for throughput, but its
                    # solution must still satisfy dynamics and cone feasibility.
                    self.assertLess(solution_metrics["r_eom"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_kinematics"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_cts_joints"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_v_plus"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_ncp_primal"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_ncp_dual"], 1.0e-2, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_ncp_compl"], 1.0e-2, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_vi_natmap"], 1.0e-2, msg=str(solution_metrics))
