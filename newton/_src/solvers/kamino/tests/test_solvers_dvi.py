# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Kamino DVI solver."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.kinematics.constraints import unpack_constraint_solutions, update_constraints_info
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians
from newton._src.solvers.kamino._src.linalg import LLTBlockedSolver
from newton._src.solvers.kamino._src.models.builders import basics, testing
from newton._src.solvers.kamino._src.models.builders import utils as builder_utils
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.padmm.types import PADMMWarmStartMode
from newton._src.solvers.kamino._src.utils.benchmark.configs import (
    make_benchmark_configs,
    make_dvi_padmm_benchmark_configs,
)
from newton._src.solvers.kamino._src.utils.benchmark.dvi_padmm_matrix import (
    make_matrix_scenarios,
    make_selected_solver_configs,
)
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import extract_delassus, extract_problem_vector
from newton._src.solvers.kamino.tests.utils.make import make_containers, make_test_problem_fourbar, update_containers


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
        testcase.assertLessEqual(int(status[wid]["iterations"]), solver.config[wid].max_iterations)
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


def _solve_dvi(model, problem, warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE) -> DVISolver:
    solver = DVISolver(
        model=model,
        config=kamino_config.DVISolverConfig(max_iterations=300, tolerance=1e-4, regularization=1e-5),
        warmstart=warmstart,
    )
    solver.reset()
    solver.coldstart()
    solver.solve(problem)
    wp.synchronize()
    return solver


def _assert_solver_status_converged(testcase: unittest.TestCase, solver: DVISolver):
    status = solver.data.status.numpy()
    for wid in range(solver.size.num_worlds):
        testcase.assertEqual(int(status[wid]["converged"]), 1)
        testcase.assertLessEqual(int(status[wid]["iterations"]), solver.config[wid].max_iterations)
        testcase.assertLessEqual(float(status[wid]["r_p"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_d"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_c"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_b"]), solver.config[wid].tolerance)


def _assert_solution_finite(testcase: unittest.TestCase, solver: DVISolver):
    testcase.assertTrue(np.all(np.isfinite(solver.data.solution.lambdas.numpy())))
    testcase.assertTrue(np.all(np.isfinite(solver.data.solution.v_plus.numpy())))


class TestDVISolver(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_00_config_selection(self):
        config = SolverKamino.Config(
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(max_iterations=32, tolerance=1e-4),
        )
        self.assertEqual(config.dynamics_solver, "dvi")
        self.assertEqual(config.dvi.max_iterations, 32)
        self.assertEqual(config.dvi.block_iterations, 8)
        self.assertEqual(config.dvi.contact_warmstart_method, "geom_pair_net_force")

        with self.assertRaises(ValueError):
            SolverKamino.Config(dynamics_solver="dvi", sparse_dynamics=True, sparse_jacobian=True)
        with self.assertRaises(ValueError):
            kamino_config.DVISolverConfig(block_iterations=0)

    def test_01_dvi_solve_dense_dual_problem(self):
        builder = basics.build_boxes_fourbar()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=0,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=None,
            jacobians=jacobians,
        )

        dynamics_config = kamino_config.ConstrainedDynamicsConfig(preconditioning=True)
        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            config=DualProblem.Config(dynamics=dynamics_config),
            solver=LLTBlockedSolver,
            sparse=False,
        )
        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)

        solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(max_iterations=1000, tolerance=1e-5, omega=1.0),
            warmstart=PADMMWarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        solver.solve(problem)
        wp.synchronize()

        _check_solution_matches_dual_problem(self, problem, solver)

    def test_02_public_solver_step_with_dvi(self):
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        builder.begin_world()
        body = builder.add_link(
            label="link",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            lock_inertia=True,
        )
        builder.add_shape_box(label="box", body=body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(
            label="hinge",
            parent=-1,
            child=body,
            axis=newton.Axis.Y,
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )
        builder.add_articulation([joint])
        builder.end_world()
        model = builder.finalize(device=self.device)

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(max_iterations=500, tolerance=1e-5),
        )
        solver = SolverKamino(model, config=config)
        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, control=None, contacts=None, dt=1e-3)
        wp.synchronize()

        body_q = state_out.body_q.numpy()
        body_qd = state_out.body_qd.numpy()
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(body_qd)))
        self.assertIsInstance(solver._solver_kamino.solver_fd, DVISolver)

    def test_03_dvi_solve_single_contact(self):
        builder = basics.build_box_on_plane()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=1,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertGreater(int(detector.contacts.model_active_contacts.numpy()[0]), 0)

        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            solver=LLTBlockedSolver,
            sparse=False,
        )
        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)

        solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(max_iterations=200, tolerance=1e-4),
            warmstart=PADMMWarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        solver.solve(problem)
        wp.synchronize()

        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 1)
        self.assertLessEqual(int(status["iterations"]), solver.config[0].max_iterations)
        self.assertTrue(np.all(np.isfinite(solver.data.solution.lambdas.numpy())))
        self.assertTrue(np.all(np.isfinite(solver.data.solution.v_plus.numpy())))

    def test_04_dvi_solve_active_joint_limit(self):
        builder = testing.build_unary_revolute_joint_test(limits=True, ground=False)
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=0,
            sparse=False,
        )
        update_containers(model=model, data=data, state=state, limits=limits, detector=None, jacobians=None)

        q_j = data.joints.q_j.numpy()
        q_j[:] = 1.0
        data.joints.q_j.assign(q_j)
        limits.detect(q_j=data.joints.q_j)
        wp.synchronize()
        update_constraints_info(model=model, data=data)
        wp.synchronize()
        jacobians.build(model=model, data=data, limits=limits.data, contacts=None)
        wp.synchronize()
        self.assertGreater(int(limits.model_active_limits.numpy()[0]), 0)

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = _solve_dvi(model, problem)

        self.assertIsNone(solver.data.unilateral_operator)
        _assert_solver_status_converged(self, solver)
        _check_solution_matches_dual_problem(self, problem, solver)

    def test_05_dvi_solve_multi_world_contacts(self):
        builder = builder_utils.make_homogeneous_builder(
            num_worlds=4,
            build_fn=basics.build_box_on_plane,
            ground=True,
        )
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertTrue(np.all(detector.contacts.world_active_contacts.numpy() > 0))

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = _solve_dvi(model, problem)

        _assert_solver_status_converged(self, solver)
        _check_solution_matches_dual_problem(self, problem, solver)

    def test_06_dvi_warmstart_modes(self):
        builder = basics.build_box_on_plane()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)

        internal_solver = _solve_dvi(model, problem, warmstart=PADMMWarmStartMode.INTERNAL)
        cold_iterations = int(internal_solver.data.status.numpy()[0]["iterations"])
        _assert_solver_status_converged(self, internal_solver)

        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)
        internal_solver.warmstart(problem, model, data)
        internal_solver.solve(problem)
        wp.synchronize()
        _assert_solver_status_converged(self, internal_solver)
        self.assertLessEqual(int(internal_solver.data.status.numpy()[0]["iterations"]), cold_iterations)

        unpack_constraint_solutions(
            lambdas=internal_solver.data.solution.lambdas,
            v_plus=internal_solver.data.solution.v_plus,
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
        )
        container_solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(max_iterations=300, tolerance=1e-4, regularization=1e-5),
            warmstart=PADMMWarmStartMode.CONTAINERS,
        )
        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)
        container_solver.warmstart(problem, model, data, limits, detector.contacts)
        container_solver.solve(problem)
        wp.synchronize()
        _assert_solver_status_converged(self, container_solver)
        _check_solution_matches_dual_problem(self, problem, container_solver)

    def test_07_dvi_singular_limit_rows_remain_finite(self):
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.device,
            max_world_contacts=0,
            with_limits=True,
            with_contacts=False,
        )
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        jacobians.build(model=model, data=data, limits=limits.data, contacts=None)
        wp.synchronize()
        self.assertGreater(int(limits.model_active_limits.numpy()[0]), 0)

        problem = _make_dense_dual_problem(model, data, limits, contacts, jacobians)
        solver = _solve_dvi(model, problem)

        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 0)
        _assert_solution_finite(self, solver)
        lambdas_np = extract_problem_vector(
            problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=True
        )[0]
        limit_start = int(problem.data.lcgo.numpy()[0])
        limit_count = int(problem.data.nl.numpy()[0])
        limit_lambdas = lambdas_np[limit_start : limit_start + limit_count]
        self.assertLess(float(np.max(np.abs(limit_lambdas))), 1.0)

    def test_08_public_solver_short_rollout_with_dvi(self):
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        builder.begin_world()
        body = builder.add_link(
            label="link",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            lock_inertia=True,
        )
        builder.add_shape_box(label="box", body=body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(
            label="hinge",
            parent=-1,
            child=body,
            axis=newton.Axis.Y,
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )
        builder.add_articulation([joint])
        builder.end_world()
        model = builder.finalize(device=self.device)

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(max_iterations=300, tolerance=1e-4),
        )
        solver = SolverKamino(model, config=config)
        state_in = model.state()
        state_out = model.state()
        for _ in range(8):
            solver.step(state_in, state_out, control=None, contacts=None, dt=1e-3)
            state_in, state_out = state_out, state_in
        wp.synchronize()

        self.assertTrue(np.all(np.isfinite(state_in.body_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_in.body_qd.numpy())))
        self.assertIsInstance(solver._solver_kamino.solver_fd, DVISolver)

    def test_09_benchmark_configs_include_dvi_dr_legs(self):
        configs = make_benchmark_configs(include_default=False)
        self.assertIn("Dense DVI Dr Legs", configs)
        config = configs["Dense DVI Dr Legs"]
        self.assertEqual(config.dynamics_solver, "dvi")
        self.assertEqual(config.integrator, "moreau")
        self.assertFalse(config.sparse_jacobian)
        self.assertFalse(config.sparse_dynamics)
        self.assertEqual(config.dvi.warmstart_mode, "containers")
        self.assertEqual(config.dvi.block_iterations, 8)

        focused_configs = make_dvi_padmm_benchmark_configs()
        self.assertEqual(set(focused_configs), {"PADMM accurate", "PADMM fast", "DVI"})
        self.assertEqual(focused_configs["DVI"].dynamics_solver, "dvi")
        self.assertEqual(focused_configs["PADMM fast"].dynamics_solver, "padmm")

    def test_10_dvi_padmm_matrix_planning(self):
        scenarios = make_matrix_scenarios(
            problem="dr_legs",
            world_counts=[1, 4],
            contact_states=["contact", "no-contact"],
            cuda_graph_modes=["off", "on"],
        )
        self.assertEqual(len(scenarios), 8)
        self.assertEqual(scenarios[0].name, "dr_legs/contact/worlds=1/graph=off")
        self.assertTrue(scenarios[0].ground)
        self.assertTrue(scenarios[1].ground)
        self.assertTrue(scenarios[1].use_cuda_graph)
        self.assertFalse(scenarios[2].ground)

        selected_configs = make_selected_solver_configs(["dvi", "padmm-fast"])
        self.assertEqual(list(selected_configs), ["DVI", "PADMM fast"])

        with self.assertRaises(ValueError):
            make_matrix_scenarios("dr_legs", [0], ["contact"], ["off"])


if __name__ == "__main__":
    unittest.main()
