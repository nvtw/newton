# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Kamino DVI solver tests."""

from __future__ import annotations
import unittest
import numpy as np
import warp as wp
import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.dynamics.dual import (
    DualProblem,
    DualProblemConfigStruct,
    _build_free_velocity_bias_contacts,
)
from newton._src.solvers.kamino._src.integrators.euler import integrate_euler_semi_implicit
from newton._src.solvers.kamino._src.kinematics.constraints import unpack_constraint_solutions, update_constraints_info
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians
from newton._src.solvers.kamino._src.linalg import LLTBlockedSolver
from newton._src.solvers.kamino._src.models.builders import basics, testing
from newton._src.solvers.kamino._src.models.builders import utils as builder_utils
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.dvi.kernels import _color_dvi_contacts, _initialize_dvi_status
from newton._src.solvers.kamino._src.solvers.dvi.sparse import (
    _SPARSE_DELASSUS_ROWS_JOINTS,
    _SPARSE_DELASSUS_ROWS_UNILATERAL,
    _sparse_delassus_matvec_rows,
)
from newton._src.solvers.kamino._src.solvers.metrics import SolutionMetrics
from newton._src.solvers.kamino._src.solvers.padmm.types import PADMMWarmStartMode
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_solvers_padmm import TestSetup
from newton._src.solvers.kamino.tests.utils.extract import extract_delassus, extract_problem_vector
from newton._src.solvers.kamino.tests.utils.make import make_containers, make_test_problem_fourbar, update_containers
from newton.tests.utils import basics as public_basics

vec2f = wp.vec2f
vec4f = wp.vec4f


def _reduce_solver_status(status: np.ndarray) -> dict[str, object]:
    """Reduce per-world status while requiring every world to converge."""
    return {
        name: bool(np.all(status[name])) if name == "converged" else np.max(status[name]).item()
        for name in status.dtype.names
    }
def _check_solution_matches_dual_problem(testcase: unittest.TestCase, problem: DualProblem, solver: DVISolver):
    """Check that final physical solution vectors match ``D lambda + v_f``."""
    D_np = extract_delassus(problem.delassus, only_active_dims=True)
    v_f_np = extract_problem_vector(problem.delassus, problem.data.v_f.numpy(), only_active_dims=True)
    P_np = extract_problem_vector(problem.delassus, problem.data.P.numpy(), only_active_dims=True)
    lambdas_np = extract_problem_vector(problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=True)
    v_plus_np = extract_problem_vector(problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=True)

    status = solver.data.status.numpy()
    for wid in range(problem.data.num_worlds):
        P_inv = np.diag(np.reciprocal(P_np[wid]))
        D_true = P_inv @ D_np[wid] @ P_inv
        v_f_true = P_inv @ v_f_np[wid]
        v_plus_true = D_true @ lambdas_np[wid] + v_f_true
        np.testing.assert_allclose(v_plus_np[wid], v_plus_true, rtol=1e-4, atol=1e-4)

        testcase.assertEqual(int(status[wid]["converged"]), 1)
        testcase.assertLessEqual(int(status[wid]["iterations"]), _status_iteration_budget(solver, wid))
        testcase.assertLessEqual(float(status[wid]["r_p"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_d"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_c"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_b"]), solver.config[wid].tolerance)
def _make_dense_dual_problem(model, data, limits, contacts, jacobians) -> DualProblem:
    problem = DualProblem(
        model=model,
        data=data,
        limits=limits,
        contacts=contacts,
        jacobians=jacobians,
        solver=LLTBlockedSolver,
        sparse=False,
    )
    problem.build(model=model, data=data, limits=limits, contacts=contacts, jacobians=jacobians)
    return problem
def _make_sparse_dual_problem(model, data, limits, contacts, jacobians) -> DualProblem:
    problem = DualProblem(
        model=model,
        data=data,
        limits=limits,
        contacts=contacts,
        jacobians=jacobians,
        sparse=True,
    )
    problem.build(model=model, data=data, limits=limits, contacts=contacts, jacobians=jacobians)
    return problem
def _solve_dvi(
    model,
    problem,
    warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE,
    config: kamino_config.DVISolverConfig | None = None,
) -> DVISolver:
    solver = DVISolver(
        model=model,
        config=config or kamino_config.DVISolverConfig(max_iterations=300, tolerance=1e-4, regularization=1e-5),
        warmstart=warmstart,
    )
    solver.reset()
    solver.coldstart()
    solver.solve(problem)
    return solver
def _status_iteration_budget(solver: DVISolver, wid: int) -> int:
    config = solver.config[wid]
    if solver._bilateral_solver is not None and solver.data.bilateral_operator is not None:
        return max(config.max_iterations, config.block_iterations * config.contact_iterations)
    return config.max_iterations
def _assert_solver_status_converged(testcase: unittest.TestCase, solver: DVISolver):
    status = solver.data.status.numpy()
    for wid in range(solver.size.num_worlds):
        testcase.assertEqual(int(status[wid]["converged"]), 1)
        testcase.assertLessEqual(int(status[wid]["iterations"]), _status_iteration_budget(solver, wid))
        testcase.assertLessEqual(float(status[wid]["r_p"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_d"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_c"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_b"]), solver.config[wid].tolerance)
def _assert_solution_finite(testcase: unittest.TestCase, solver: DVISolver):
    testcase.assertTrue(np.all(np.isfinite(solver.data.solution.lambdas.numpy())))
    testcase.assertTrue(np.all(np.isfinite(solver.data.solution.v_plus.numpy())))
def _evaluate_solution_metrics(test: TestSetup, solver: DVISolver) -> dict[str, float]:
    integrate_euler_semi_implicit(model=test.model, data=test.data)
    metrics = SolutionMetrics(model=test.model)
    metrics.evaluate(
        sigma=solver.data.state.sigma,
        lambdas=solver.data.solution.lambdas,
        v_plus=solver.data.solution.v_plus,
        model=test.model,
        data=test.data,
        state_p=test.state_p,
        problem=test.problem,
        jacobians=test.jacobians,
        limits=test.limits,
        contacts=test.contacts,
    )
    return {
        name: float(np.max(getattr(metrics.data, name).numpy()))
        for name in (
            "r_eom",
            "r_kinematics",
            "r_cts_joints",
            "r_cts_limits",
            "r_cts_contacts",
            "r_v_plus",
            "r_ncp_primal",
            "r_ncp_dual",
            "r_ncp_compl",
            "r_vi_natmap",
        )
    }


class DVITestCase(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

__all__ = (
    "unittest",
    "np",
    "wp",
    "newton",
    "kamino_config",
    "DualProblem",
    "DualProblemConfigStruct",
    "_build_free_velocity_bias_contacts",
    "integrate_euler_semi_implicit",
    "unpack_constraint_solutions",
    "update_constraints_info",
    "DenseSystemJacobians",
    "LLTBlockedSolver",
    "basics",
    "testing",
    "builder_utils",
    "DVISolver",
    "_color_dvi_contacts",
    "_initialize_dvi_status",
    "_SPARSE_DELASSUS_ROWS_JOINTS",
    "_SPARSE_DELASSUS_ROWS_UNILATERAL",
    "_sparse_delassus_matvec_rows",
    "SolutionMetrics",
    "PADMMWarmStartMode",
    "SolverKamino",
    "setup_tests",
    "test_context",
    "TestSetup",
    "extract_delassus",
    "extract_problem_vector",
    "make_containers",
    "make_test_problem_fourbar",
    "update_containers",
    "public_basics",
    "vec2f",
    "vec4f",
    "_reduce_solver_status",
    "_check_solution_matches_dual_problem",
    "_make_dense_dual_problem",
    "_make_sparse_dual_problem",
    "_solve_dvi",
    "_status_iteration_budget",
    "_assert_solver_status_converged",
    "_assert_solution_finite",
    "_evaluate_solution_metrics",
    "DVITestCase",
)
