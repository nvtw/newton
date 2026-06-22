# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Kamino DVI solver."""

from __future__ import annotations

import contextlib
import unittest

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.types import vec2f, vec4f
from newton._src.solvers.kamino._src.dynamics.dual import (
    DualProblem,
    DualProblemConfigStruct,
    _build_free_velocity_bias_contacts,
)
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
from newton._src.solvers.kamino._src.solvers.padmm.types import PADMMWarmStartMode
from newton._src.solvers.kamino._src.utils.benchmark.configs import (
    load_solver_configs_to_hdf5,
    make_benchmark_configs,
    make_dvi_padmm_benchmark_configs,
    save_solver_configs_to_hdf5,
)
from newton._src.solvers.kamino._src.utils.benchmark.dvi_padmm_matrix import (
    apply_dvi_overrides,
    make_matrix_scenarios,
    make_selected_solver_configs,
)
from newton._src.solvers.kamino._src.utils.benchmark.render import render_solver_configs_table
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
    wp.synchronize()
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


class _MiniHDF5Dataset:
    def __init__(self, value):
        self._value = value

    def __getitem__(self, key):
        if key != ():
            raise KeyError(key)
        return self._value


class _MiniHDF5Group:
    def __init__(self, values: dict[str, object], prefix: str):
        self._values = values
        self._prefix = f"{prefix}/"

    def keys(self):
        names = set()
        for path in self._values:
            if path.startswith(self._prefix):
                names.add(path.removeprefix(self._prefix).split("/", 1)[0])
        return names


class _MiniHDF5File:
    def __init__(self):
        self._values: dict[str, object] = {}

    def __contains__(self, path: str):
        return path in self._values or any(key.startswith(f"{path}/") for key in self._values)

    def __getitem__(self, path: str):
        if path in self._values:
            return _MiniHDF5Dataset(self._values[path])
        if path in self:
            return _MiniHDF5Group(self._values, path)
        raise KeyError(path)

    def __setitem__(self, path: str, value):
        self._values[path] = value


class _EncodingCheckingStdout:
    encoding = "cp1252"

    def __init__(self):
        self.output: list[str] = []

    def write(self, text: str) -> int:
        text.encode(self.encoding)
        self.output.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True


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
        self.assertEqual(config.dvi.block_iterations, 32)
        self.assertEqual(config.dvi.contact_iterations, 4)
        self.assertEqual(config.dvi.bilateral_solve_period, 1)
        self.assertEqual(config.dvi.contact_jacobi_omega, 0.3)
        self.assertEqual(config.dvi.contact_jacobi_relaxation, 0.9)
        self.assertFalse(config.dvi.contact_block_preconditioner)
        self.assertEqual(config.dvi.contact_warmstart_method, "geom_pair_net_force")
        self.assertFalse(config.dynamics.preconditioning)
        self.assertEqual(config.constraints.contact_recovery_speed, -1.0)

        sparse_config = SolverKamino.Config(dynamics_solver="dvi", sparse_dynamics=True, sparse_jacobian=True)
        self.assertTrue(sparse_config.sparse_dynamics)
        self.assertTrue(sparse_config.sparse_jacobian)
        with self.assertRaises(ValueError):
            SolverKamino.Config(
                dynamics_solver="dvi",
                dynamics=kamino_config.ConstrainedDynamicsConfig(preconditioning=True),
            )
        with self.assertRaises(ValueError):
            kamino_config.DVISolverConfig(block_iterations=0)
        with self.assertRaises(ValueError):
            kamino_config.DVISolverConfig(contact_iterations=0)
        with self.assertRaises(ValueError):
            kamino_config.DVISolverConfig(bilateral_solve_period=0)
        with self.assertRaises(ValueError):
            kamino_config.DVISolverConfig(contact_jacobi_omega=0.0)
        with self.assertRaises(ValueError):
            kamino_config.DVISolverConfig(contact_jacobi_relaxation=1.1)

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
        self.assertFalse(solver._solver_kamino.config.dynamics.preconditioning)

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
        self.assertLessEqual(int(status["iterations"]), _status_iteration_budget(solver, 0))
        self.assertTrue(np.all(np.isfinite(solver.data.solution.lambdas.numpy())))
        self.assertTrue(np.all(np.isfinite(solver.data.solution.v_plus.numpy())))

    def test_03a_sparse_dvi_filtered_matvec_matches_full_rows(self):
        builder = basics.build_box_on_plane()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=True,
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

        problem = _make_sparse_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                block_iterations=1,
                contact_iterations=1,
            ),
            warmstart=PADMMWarmStartMode.NONE,
        )
        solver.reset()

        lambdas = np.linspace(-0.25, 0.5, problem.data.v_f.shape[0], dtype=np.float32)
        solver.data.solution.lambdas.assign(lambdas)

        full = wp.zeros_like(problem.data.v_f)
        problem.delassus.matvec(solver.data.solution.lambdas, full, solver.data.state.world_mask)
        full_np = full.numpy()

        _sparse_delassus_matvec_rows(solver, problem, _SPARSE_DELASSUS_ROWS_JOINTS)
        joint_np = solver.data.state.v_aug.numpy()
        _sparse_delassus_matvec_rows(solver, problem, _SPARSE_DELASSUS_ROWS_UNILATERAL)
        unilateral_np = solver.data.state.v_aug.numpy()

        dim = int(problem.data.dim.numpy()[0])
        njc = int(problem.data.njc.numpy()[0])
        np.testing.assert_allclose(joint_np[:njc], full_np[:njc], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(unilateral_np[njc:dim], full_np[njc:dim], rtol=1e-5, atol=1e-5)

    def test_03b_dvi_contact_block_preconditioner_smoke(self):
        builder = basics.build_boxes_hinged()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=8,
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

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)

        def solve_with_block_preconditioner(use_colored_contacts: bool) -> DVISolver:
            solver = DVISolver(
                model=model,
                config=kamino_config.DVISolverConfig(
                    max_iterations=200,
                    tolerance=1e-4,
                    contact_block_preconditioner=True,
                ),
                warmstart=PADMMWarmStartMode.NONE,
            )
            solver.reset()
            solver.coldstart()
            if use_colored_contacts:
                solver.set_contacts(detector.contacts)
            solver.solve(problem)
            wp.synchronize()
            return solver

        contact_paths = [False]
        if self.device.is_cuda:
            contact_paths.append(True)
        for use_colored_contacts in contact_paths:
            with self.subTest(use_colored_contacts=use_colored_contacts):
                solver = solve_with_block_preconditioner(use_colored_contacts)

                if use_colored_contacts:
                    self.assertGreater(int(solver.data.state.contact_num_colors.numpy()[0]), 0)
                else:
                    self.assertEqual(int(solver.data.state.contact_num_colors.numpy()[0]), 0)
                _assert_solution_finite(self, solver)
                _check_solution_matches_dual_problem(self, problem, solver)

    def test_03c_dvi_noncolored_contact_jacobi_uses_configured_omega(self):
        builder = basics.build_boxes_hinged()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=8,
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

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)

        def solve_normal_lambda(contact_jacobi_omega: float) -> float:
            solver = DVISolver(
                model=model,
                config=kamino_config.DVISolverConfig(
                    tolerance=0.0,
                    regularization=1e-5,
                    block_iterations=1,
                    contact_iterations=1,
                    contact_jacobi_omega=contact_jacobi_omega,
                    contact_jacobi_relaxation=1.0,
                ),
                warmstart=PADMMWarmStartMode.NONE,
            )
            solver.reset()
            solver.coldstart()
            solver.solve(problem)
            wp.synchronize()

            self.assertEqual(int(solver.data.state.contact_num_colors.numpy()[0]), 0)
            lambdas = extract_problem_vector(
                problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=True
            )[0]
            contact_start = int(problem.data.ccgo.numpy()[0])
            contact_count = int(problem.data.nc.numpy()[0])
            normal_lambdas = lambdas[contact_start + 2 : contact_start + 3 * contact_count : 3]
            return float(np.sum(normal_lambdas))

        lambda_slow = solve_normal_lambda(0.1)
        lambda_fast = solve_normal_lambda(0.8)

        self.assertTrue(np.isfinite(lambda_slow))
        self.assertTrue(np.isfinite(lambda_fast))
        self.assertGreater(lambda_fast, lambda_slow)

    def test_03d_dvi_direct_block_honors_per_world_iteration_counts(self):
        builder = builder_utils.make_homogeneous_builder(
            num_worlds=3,
            build_fn=basics.build_boxes_hinged,
        )
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=8,
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
        configs = [
            kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                block_iterations=1,
                contact_iterations=1,
                contact_jacobi_relaxation=1.0,
            ),
            kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                block_iterations=3,
                contact_iterations=1,
                contact_jacobi_relaxation=1.0,
            ),
            kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                block_iterations=1,
                contact_iterations=3,
                contact_jacobi_relaxation=1.0,
            ),
        ]

        def solve_normal_sums(use_colored_contacts: bool) -> list[float]:
            solver = DVISolver(model=model, config=configs, warmstart=PADMMWarmStartMode.NONE)
            solver.reset()
            solver.coldstart()
            if use_colored_contacts:
                solver.set_contacts(detector.contacts)
            solver.solve(problem)
            wp.synchronize()

            status = solver.data.status.numpy()
            self.assertEqual([int(status[wid]["iterations"]) for wid in range(3)], [1, 3, 3])
            if use_colored_contacts:
                self.assertTrue(np.all(solver.data.state.contact_num_colors.numpy() > 0))
            else:
                self.assertTrue(np.all(solver.data.state.contact_num_colors.numpy() == 0))
            np.testing.assert_array_equal(
                solver.data.state.bilateral_active_dim.numpy(),
                problem.data.njc.numpy(),
            )

            lambdas = extract_problem_vector(
                problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=True
            )
            ccgo = problem.data.ccgo.numpy().astype(int)
            nc = problem.data.nc.numpy().astype(int)
            return [float(np.sum(lambdas[wid][ccgo[wid] + 2 : ccgo[wid] + 3 * nc[wid] : 3])) for wid in range(3)]

        contact_paths = [False]
        if self.device.is_cuda:
            contact_paths.append(True)
        for use_colored_contacts in contact_paths:
            with self.subTest(use_colored_contacts=use_colored_contacts):
                normal_sums = solve_normal_sums(use_colored_contacts)

                self.assertGreater(normal_sums[1], normal_sums[0])
                self.assertGreater(normal_sums[2], normal_sums[0])

    def test_03d2_dvi_direct_block_finishes_with_bilateral_solve(self):
        builder = basics.build_boxes_hinged()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=8,
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
        self.assertGreater(int(detector.contacts.world_active_contacts.numpy()[0]), 0)

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                block_iterations=1,
                contact_iterations=1,
                contact_jacobi_relaxation=1.0,
            ),
            warmstart=PADMMWarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        solver.solve(problem)
        wp.synchronize()

        v_plus = extract_problem_vector(problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=True)[0]
        njc = int(problem.data.njc.numpy()[0])
        status = solver.data.status.numpy()[0]

        self.assertGreater(njc, 0)
        self.assertLess(float(np.max(np.abs(v_plus[:njc]))), 1e-6)
        self.assertLess(float(status["r_b"]), 1e-6)

    def test_03e_dvi_direct_block_no_unilateral_rows_reports_single_iteration(self):
        builder = basics.build_box_pendulum(ground=False)
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
        self.assertGreater(int(model.info.num_joint_cts.numpy()[0]), 0)
        self.assertEqual(int(problem.data.nl.numpy()[0]), 0)
        self.assertEqual(int(problem.data.nc.numpy()[0]), 0)

        solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(
                tolerance=1e-4,
                regularization=1e-5,
                block_iterations=7,
                contact_iterations=3,
            ),
            warmstart=PADMMWarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        solver.solve(problem)
        wp.synchronize()

        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 1)
        self.assertEqual(int(status["iterations"]), 1)
        self.assertEqual(int(solver.data.state.bilateral_active_dim.numpy()[0]), 0)
        _check_solution_matches_dual_problem(self, problem, solver)

    def test_03f_dvi_bilateral_only_solve_resets_stale_status(self):
        builder = basics.build_box_pendulum(ground=False)
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

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        self.assertGreater(int(model.info.num_joint_cts.numpy()[0]), 0)
        self.assertEqual(int(problem.data.nl.numpy()[0]), 0)
        self.assertEqual(int(problem.data.nc.numpy()[0]), 0)

        solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(
                tolerance=1e-4,
                regularization=1e-5,
                contact_iterations=5,
            ),
            warmstart=PADMMWarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        wp.launch(
            kernel=_initialize_dvi_status,
            dim=solver.size.num_worlds,
            inputs=[
                solver.data.config,
                solver.data.status,
            ],
            device=self.device,
        )
        wp.synchronize()
        self.assertEqual(int(solver.data.status.numpy()[0]["iterations"]), 5)

        solver.solve(problem)
        wp.synchronize()

        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 1)
        self.assertEqual(int(status["iterations"]), 1)
        _check_solution_matches_dual_problem(self, problem, solver)

    def test_03g_dvi_contact_coloring_separates_dynamic_conflicts(self):
        problem_nc = wp.array([5], dtype=wp.int32, device=self.device)
        problem_cio = wp.array([0], dtype=wp.int32, device=self.device)
        contact_bid_ab = wp.array(
            [
                wp.vec2i(0, -1),
                wp.vec2i(0, 1),
                wp.vec2i(2, -1),
                wp.vec2i(-1, -1),
                wp.vec2i(1, -1),
            ],
            dtype=wp.vec2i,
            device=self.device,
        )
        contact_colors = wp.full(shape=5, value=-1, dtype=wp.int32, device=self.device)
        contact_num_colors = wp.zeros(shape=1, dtype=wp.int32, device=self.device)

        wp.launch(
            kernel=_color_dvi_contacts,
            dim=1,
            inputs=[
                problem_nc,
                problem_cio,
                contact_bid_ab,
                contact_colors,
                contact_num_colors,
            ],
            device=self.device,
        )
        wp.synchronize()

        colors = contact_colors.numpy()
        num_colors = int(contact_num_colors.numpy()[0])
        self.assertGreaterEqual(num_colors, 2)
        self.assertTrue(np.all(colors >= 0))
        self.assertNotEqual(colors[0], colors[1])
        self.assertNotEqual(colors[1], colors[4])
        self.assertLess(colors[3], num_colors)

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
        solver = _solve_dvi(
            model,
            problem,
            config=kamino_config.DVISolverConfig(max_iterations=8, tolerance=1e-4, regularization=1e-5),
        )

        _assert_solver_status_converged(self, solver)
        iterations = int(solver.data.status.numpy()[0]["iterations"])
        self.assertEqual(iterations, solver.config[0].block_iterations * solver.config[0].contact_iterations)
        self.assertGreater(iterations, solver.config[0].max_iterations)
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

    def test_09_dr_legs_dvi_first_contact_remains_finite(self):
        if not self.device.is_cuda:
            self.skipTest("Dr Legs DVI first-contact regression uses the CUDA graph path")

        from types import SimpleNamespace  # noqa: PLC0415

        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        args = SimpleNamespace(
            world_count=1,
            use_kamino_contacts=True,
            dynamics_solver="dvi",
            dvi_contact_block_preconditioner=False,
            dvi_contact_jacobi_omega=0.25,
            dvi_contact_jacobi_relaxation=0.9,
        )
        example = Example(ViewerNull(num_frames=1), args)

        contact_seen = False
        color_checked = False
        for _ in range(12):
            example.step()
            body_q = example.state_0.body_q.numpy()
            body_qd = example.state_0.body_qd.numpy()
            lambdas = example.solver._solver_kamino.solver_fd.data.solution.lambdas.numpy()
            kamino_contacts = example.solver._contacts_kamino
            contact_count = int(kamino_contacts.world_active_contacts.numpy()[0])
            contact_seen = contact_seen or contact_count > 0
            if contact_count > 0 and not example.config.sparse_dynamics:
                solver_fd = example.solver._solver_kamino.solver_fd
                color_count = int(solver_fd.data.state.contact_num_colors.numpy()[0])
                colors = solver_fd.data.state.contact_colors.numpy()
                bid_ab = kamino_contacts.bid_AB.numpy()
                self.assertGreater(color_count, 0)
                self.assertTrue(np.all(colors[:contact_count] >= 0))
                for ci in range(contact_count):
                    bodies_i = {int(bid_ab[ci][0]), int(bid_ab[ci][1])} - {-1}
                    for cj in range(ci):
                        if colors[ci] == colors[cj]:
                            bodies_j = {int(bid_ab[cj][0]), int(bid_ab[cj][1])} - {-1}
                            self.assertFalse(bodies_i & bodies_j)
                color_checked = True
            elif contact_count > 0:
                color_checked = True

            self.assertTrue(np.all(np.isfinite(body_q)))
            self.assertTrue(np.all(np.isfinite(body_qd)))
            self.assertTrue(np.all(np.isfinite(lambdas)))
            self.assertLess(float(np.max(np.abs(body_qd))), 100.0)
            self.assertLess(float(np.max(np.abs(lambdas))), 100.0)

        self.assertTrue(contact_seen)
        self.assertTrue(color_checked)

    def test_10_dr_legs_dvi_tipped_contact_does_not_creep(self):
        if not self.device.is_cuda:
            self.skipTest("Dr Legs DVI tipped-contact regression uses the CUDA graph path")

        from types import SimpleNamespace  # noqa: PLC0415

        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        args = SimpleNamespace(
            world_count=1,
            use_kamino_contacts=True,
            dynamics_solver="dvi",
            dvi_contact_block_preconditioner=False,
            dvi_contact_jacobi_omega=0.25,
            dvi_contact_jacobi_relaxation=0.9,
        )
        example = Example(ViewerNull(num_frames=1), args)

        q_tip = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.pi * 0.5))
        example.base_q.assign([wp.transformf((0.0, 0.0, 0.25), q_tip)])
        example.solver.reset(state=example.state_0, base_q=example.base_q)
        example.capture()

        base_start = example.state_0.body_q.numpy()[0, :3].copy()
        contact_seen = False
        for _ in range(160):
            example.step()
            contact_seen = contact_seen or int(example.contacts.rigid_contact_count.numpy()[0]) > 0

        body_q = example.state_0.body_q.numpy()
        body_qd = example.state_0.body_qd.numpy()
        base_delta_xy = body_q[0, :2] - base_start[:2]

        self.assertTrue(contact_seen)
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(body_qd)))
        self.assertLess(float(np.linalg.norm(base_delta_xy)), 0.006)

    def test_11_dr_legs_dvi_contact_force_balances_weight(self):
        if not self.device.is_cuda:
            self.skipTest("Dr Legs DVI contact-force regression uses the CUDA graph path")

        from types import SimpleNamespace  # noqa: PLC0415

        from newton._src.solvers.kamino._src.geometry.aggregation import ContactAggregation  # noqa: PLC0415
        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        args = SimpleNamespace(
            world_count=1,
            use_kamino_contacts=True,
            dynamics_solver="dvi",
            dvi_contact_block_preconditioner=False,
            dvi_contact_jacobi_omega=0.25,
            dvi_contact_jacobi_relaxation=0.9,
        )
        example = Example(ViewerNull(num_frames=1), args)

        for _ in range(180):
            example.step()

        contacts_kamino = example.solver._contacts_kamino
        aggregation = ContactAggregation(model=example.solver._model_kamino, contacts=contacts_kamino)
        aggregation.compute()

        contact_count = int(contacts_kamino.world_active_contacts.numpy()[0])
        total_contact_force = aggregation.body_net_force.numpy()[0].sum(axis=0)
        weight = float(example.model.body_mass.numpy().sum() * 9.81)
        force_ratio = float(total_contact_force[2] / weight)

        self.assertGreater(contact_count, 0)
        self.assertTrue(np.all(np.isfinite(total_contact_force)))
        self.assertGreater(force_ratio, 0.95)
        self.assertLess(force_ratio, 1.05)

    def test_12_dvi_opening_contact_releases_warmstarted_force(self):
        if not self.device.is_cuda:
            self.skipTest("DVI colored contact release regression uses the CUDA graph-colored path")

        radius = 0.1
        separation = 0.005
        gap = 0.03
        z = radius + separation

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(gap=gap, margin=0.0)
        body = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_sphere(body=body, radius=radius, cfg=shape_cfg)
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=body,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()),
            child_xform=wp.transform_identity(),
            limit_lower=-10.0,
            limit_upper=10.0,
        )
        builder.add_articulation([joint])
        builder.add_ground_plane(cfg=shape_cfg)
        model = builder.finalize(device=self.device)

        joint_qd = model.joint_qd.numpy()
        joint_qd[:] = 1.0
        model.joint_qd.assign(joint_qd)

        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        config = SolverKamino.Config(
            use_collision_detector=True,
            collision_detector=kamino_config.CollisionDetectorConfig(
                max_contacts_per_world=8,
                max_contacts_per_pair=8,
                default_gap=gap,
            ),
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(
                max_iterations=300,
                tolerance=1e-5,
                regularization=1e-5,
                block_iterations=32,
                contact_iterations=4,
            ),
        )
        solver = SolverKamino(model, config=config)

        solver.step(state_0, state_1, control=None, contacts=None, dt=1e-3)
        self.assertEqual(int(solver._contacts_kamino.model_active_contacts.numpy()[0]), 1)

        cache = solver._solver_kamino._ws_contacts.cache
        self.assertIsNotNone(cache)
        reaction = cache.reaction.numpy()
        reaction[:, :] = 0.0
        reaction[0, 2] = 10000.0
        cache.reaction.assign(reaction)
        velocity = cache.velocity.numpy()
        velocity[:, :] = 0.0
        cache.velocity.assign(velocity)

        solver.step(state_1, state_0, control=None, contacts=None, dt=1e-3)

        contact_count = int(solver._contacts_kamino.model_active_contacts.numpy()[0])
        gaps = solver._contacts_kamino.gapfunc.numpy()[:contact_count, 3]
        contact_velocity = solver._contacts_kamino.velocity.numpy()[:contact_count, 2]
        contact_reaction = solver._contacts_kamino.reaction.numpy()[:contact_count, 2]
        opening = (gaps > 0.0) & (contact_velocity > 0.0)

        self.assertTrue(np.any(opening))
        self.assertLess(float(np.max(np.abs(contact_reaction[opening]))), 1e-3)
        self.assertLess(float(abs(state_0.body_qd.numpy()[0, 2])), 2.0)
        self.assertEqual(int(solver._solver_kamino.solver_fd.data.status.numpy()[0]["converged"]), 1)

    def test_12b_dvi_contact_recovery_speed_caps_bias(self):
        config = DualProblem.Config(
            constraints=kamino_config.ConstraintStabilizationConfig(
                gamma=1.0,
                delta=0.0,
                contact_recovery_speed=0.2,
            )
        )
        problem_config = wp.array([config.to_struct()], dtype=DualProblemConfigStruct, device=self.device)
        model_time_inv_dt = wp.array([100.0], dtype=wp.float32, device=self.device)
        model_info_contacts_offset = wp.array([0], dtype=wp.int32, device=self.device)
        data_info_contact_cts_group_offset = wp.array([0], dtype=wp.int32, device=self.device)
        contacts_model_num = wp.array([1], dtype=wp.int32, device=self.device)
        contacts_wid = wp.array([0], dtype=wp.int32, device=self.device)
        contacts_cid = wp.array([0], dtype=wp.int32, device=self.device)
        contacts_material = wp.array([vec2f(0.5, 0.0)], dtype=vec2f, device=self.device)
        problem_vio = wp.array([0], dtype=wp.int32, device=self.device)

        def contact_bias(distance: float) -> np.ndarray:
            contacts_gapfunc = wp.array([vec4f(0.0, 0.0, 0.0, distance)], dtype=vec4f, device=self.device)
            problem_v_b = wp.zeros(3, dtype=wp.float32, device=self.device)
            problem_v_i = wp.zeros(3, dtype=wp.float32, device=self.device)
            problem_mu = wp.zeros(1, dtype=wp.float32, device=self.device)
            wp.launch(
                _build_free_velocity_bias_contacts,
                dim=1,
                inputs=[
                    model_time_inv_dt,
                    model_info_contacts_offset,
                    data_info_contact_cts_group_offset,
                    1,
                    contacts_model_num,
                    contacts_wid,
                    contacts_cid,
                    contacts_gapfunc,
                    contacts_material,
                    problem_config,
                    problem_vio,
                ],
                outputs=[problem_v_b, problem_v_i, problem_mu],
                device=self.device,
            )
            return problem_v_b.numpy()

        np.testing.assert_allclose(contact_bias(-0.05), [0.0, 0.0, -0.2], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(contact_bias(0.05), [0.0, 0.0, 5.0], rtol=1e-6, atol=1e-6)

    def test_13_benchmark_configs_include_dvi_dr_legs(self):
        configs = make_benchmark_configs(include_default=False)
        self.assertIn("Dense DVI Dr Legs", configs)
        self.assertIn("Sparse DVI Dr Legs", configs)
        config = configs["Dense DVI Dr Legs"]
        self.assertEqual(config.dynamics_solver, "dvi")
        self.assertEqual(config.integrator, "moreau")
        self.assertFalse(config.sparse_jacobian)
        self.assertFalse(config.sparse_dynamics)
        self.assertEqual(config.dvi.warmstart_mode, "containers")
        self.assertEqual(config.dvi.block_iterations, 32)
        self.assertEqual(config.dvi.contact_iterations, 4)
        self.assertEqual(config.dvi.bilateral_solve_period, 1)
        self.assertEqual(config.dvi.contact_jacobi_omega, 0.3)
        self.assertEqual(config.dvi.contact_jacobi_relaxation, 0.9)
        self.assertFalse(config.dvi.contact_block_preconditioner)
        self.assertEqual(config.constraints.gamma, 0.015)
        self.assertEqual(config.constraints.delta, 1.0e-6)
        self.assertEqual(config.constraints.contact_recovery_speed, 1.0)
        self.assertFalse(config.dynamics.preconditioning)

        focused_configs = make_dvi_padmm_benchmark_configs()
        self.assertEqual(set(focused_configs), {"PADMM accurate", "PADMM fast", "DVI"})
        self.assertEqual(focused_configs["DVI"].dynamics_solver, "dvi")
        self.assertTrue(focused_configs["DVI"].sparse_jacobian)
        self.assertTrue(focused_configs["DVI"].sparse_dynamics)
        self.assertEqual(focused_configs["DVI"].dvi.block_iterations, 16)
        self.assertEqual(focused_configs["DVI"].dvi.contact_iterations, 2)
        self.assertEqual(focused_configs["DVI"].dvi.bilateral_solve_period, 2)
        self.assertEqual(focused_configs["PADMM fast"].dynamics_solver, "padmm")

    def test_13b_dvi_benchmark_config_roundtrips_contact_controls(self):
        focused_configs = make_dvi_padmm_benchmark_configs()
        config = focused_configs["DVI"]
        config.dvi.contact_block_preconditioner = True
        config.dvi.contact_jacobi_omega = 0.25
        config.dvi.contact_jacobi_relaxation = 0.75
        config.dvi.bilateral_solve_period = 2
        config.dvi.warmstart_mode = "internal"
        config.dvi.contact_warmstart_method = "reaction"

        datafile = _MiniHDF5File()
        save_solver_configs_to_hdf5({"DVI tuned": config}, datafile)
        self.assertTrue(bool(datafile["Solver/DVI tuned/dvi/contact_block_preconditioner"][()]))
        self.assertEqual(float(datafile["Solver/DVI tuned/dvi/contact_jacobi_omega"][()]), 0.25)
        self.assertEqual(float(datafile["Solver/DVI tuned/dvi/contact_jacobi_relaxation"][()]), 0.75)
        self.assertEqual(int(datafile["Solver/DVI tuned/dvi/bilateral_solve_period"][()]), 2)

        loaded_configs = load_solver_configs_to_hdf5(datafile)

        self.assertTrue(loaded_configs["DVI tuned"].dvi.contact_block_preconditioner)
        self.assertEqual(loaded_configs["DVI tuned"].dvi.contact_jacobi_omega, 0.25)
        self.assertEqual(loaded_configs["DVI tuned"].dvi.contact_jacobi_relaxation, 0.75)
        self.assertEqual(loaded_configs["DVI tuned"].dvi.bilateral_solve_period, 2)
        self.assertEqual(loaded_configs["DVI tuned"].dvi.warmstart_mode, "internal")
        self.assertEqual(loaded_configs["DVI tuned"].dvi.contact_warmstart_method, "reaction")

        datafile._values.pop("Solver/DVI tuned/dvi/warmstart_mode")
        datafile._values.pop("Solver/DVI tuned/dvi/contact_warmstart_method")
        datafile._values.pop("Solver/DVI tuned/dvi/bilateral_solve_period")
        loaded_legacy_configs = load_solver_configs_to_hdf5(datafile)
        dvi_defaults = kamino_config.DVISolverConfig()

        self.assertEqual(
            loaded_legacy_configs["DVI tuned"].dvi.bilateral_solve_period,
            dvi_defaults.bilateral_solve_period,
        )
        self.assertEqual(loaded_legacy_configs["DVI tuned"].dvi.warmstart_mode, dvi_defaults.warmstart_mode)
        self.assertEqual(
            loaded_legacy_configs["DVI tuned"].dvi.contact_warmstart_method,
            dvi_defaults.contact_warmstart_method,
        )

    def test_13c_dvi_benchmark_render_is_windows_console_safe(self):
        focused_configs = make_dvi_padmm_benchmark_configs()
        stdout = _EncodingCheckingStdout()

        with contextlib.redirect_stdout(stdout):
            render_solver_configs_table(
                configs={"DVI": focused_configs["DVI"]},
                groups=["solver", "dvi"],
                to_console=True,
            )

        self.assertGreater(len("".join(stdout.output)), 0)

    def test_14_dvi_padmm_matrix_planning(self):
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

        from types import SimpleNamespace  # noqa: PLC0415

        padmm_block_iterations = selected_configs["PADMM fast"].dvi.block_iterations
        args = SimpleNamespace(
            dvi_block_iterations=5,
            dvi_contact_iterations=6,
            dvi_bilateral_solve_period=2,
            dvi_contact_jacobi_omega=0.4,
            dvi_contact_jacobi_relaxation=0.8,
            dvi_contact_block_preconditioner=True,
        )
        apply_dvi_overrides(selected_configs, args)

        self.assertEqual(selected_configs["DVI"].dvi.block_iterations, 5)
        self.assertEqual(selected_configs["DVI"].dvi.contact_iterations, 6)
        self.assertEqual(selected_configs["DVI"].dvi.bilateral_solve_period, 2)
        self.assertEqual(selected_configs["DVI"].dvi.contact_jacobi_omega, 0.4)
        self.assertEqual(selected_configs["DVI"].dvi.contact_jacobi_relaxation, 0.8)
        self.assertTrue(selected_configs["DVI"].dvi.contact_block_preconditioner)
        self.assertEqual(selected_configs["PADMM fast"].dvi.block_iterations, padmm_block_iterations)

        with self.assertRaises(ValueError):
            make_matrix_scenarios("dr_legs", [0], ["contact"], ["off"])


if __name__ == "__main__":
    unittest.main()
