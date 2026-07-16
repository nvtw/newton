# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused Kamino DVI Dense tests."""

from newton._src.solvers.kamino.tests.dvi_test_helpers import *  # noqa: F403


class TestDVIDense(DVITestCase):
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
            collect_info=True,
        )
        solver.reset()
        scratch = solver.data.state
        for array in (
            scratch.v_aug,
            scratch.s,
            scratch.scratch,
            scratch.bilateral_rhs,
            scratch.bilateral_solution,
            scratch.bilateral_preconditioner,
        ):
            array.fill_(float("nan"))
        scratch.bilateral_active_dim.fill_(-1)
        scratch.contact_colors.fill_(-1)
        scratch.contact_num_colors.fill_(-1)
        solver.coldstart()
        solver.solve(problem)
        _check_solution_matches_dual_problem(self, problem, solver)
        np.testing.assert_array_equal(solver.data.info.status.numpy(), solver.data.status.numpy())
        status = solver.data.status.numpy()
        np.testing.assert_array_equal(solver.data.info.iterations.numpy(), status["iterations"])
        np.testing.assert_allclose(solver.data.info.r_p.numpy(), status["r_p"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(solver.data.info.r_d.numpy(), status["r_d"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(solver.data.info.r_c.numpy(), status["r_c"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(solver.data.info.r_b.numpy(), status["r_b"], rtol=0.0, atol=0.0)


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
            collect_solver_info=True,
        )
        solver = SolverKamino(model, config=config)
        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, control=None, contacts=None, dt=1e-3)
        body_q = state_out.body_q.numpy()
        body_qd = state_out.body_qd.numpy()
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(body_qd)))
        self.assertIsInstance(solver._solver_kamino.solver_fd, DVISolver)
        self.assertFalse(solver._solver_kamino.config.dynamics.preconditioning)
        self.assertIsNotNone(solver._solver_kamino.solver_fd.data.info)


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
        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 1)
        self.assertLessEqual(int(status["iterations"]), _status_iteration_budget(solver, 0))
        self.assertTrue(np.all(np.isfinite(solver.data.solution.lambdas.numpy())))
        self.assertTrue(np.all(np.isfinite(solver.data.solution.v_plus.numpy())))


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
        self.assertEqual(int(solver.data.status.numpy()[0]["iterations"]), 5)

        solver.solve(problem)
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
        colors = contact_colors.numpy()
        num_colors = int(contact_num_colors.numpy()[0])
        self.assertGreaterEqual(num_colors, 2)
        self.assertTrue(np.all(colors >= 0))
        self.assertNotEqual(colors[0], colors[1])
        self.assertNotEqual(colors[1], colors[4])
        self.assertLess(colors[3], num_colors)


    def test_03i_dvi_coldstart_is_repeatable(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                test = TestSetup(
                    builder_fn=basics.build_boxes_hinged,
                    max_world_contacts=8,
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
                first_lambdas = solver.data.solution.lambdas.numpy().copy()
                first_v_plus = solver.data.solution.v_plus.numpy().copy()
                first_status = solver.data.status.numpy().copy()

                test.build()
                solver.reset()
                solver.coldstart()
                solver.solve(test.problem)

                np.testing.assert_allclose(solver.data.solution.lambdas.numpy(), first_lambdas, rtol=0.0, atol=1e-6)
                np.testing.assert_allclose(solver.data.solution.v_plus.numpy(), first_v_plus, rtol=0.0, atol=1e-6)
                status = solver.data.status.numpy()
                np.testing.assert_array_equal(status["converged"], first_status["converged"])
                np.testing.assert_array_equal(status["iterations"], first_status["iterations"])
                for residual in ("r_p", "r_d", "r_c", "r_b"):
                    np.testing.assert_allclose(status[residual], first_status[residual], rtol=1e-5, atol=1e-8)


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
        update_constraints_info(model=model, data=data)
        jacobians.build(model=model, data=data, limits=limits.data, contacts=None)
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


    def test_07_dvi_singular_limit_rows_remain_finite(self):
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.device,
            max_world_contacts=0,
            with_limits=True,
            with_contacts=False,
        )
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        jacobians.build(model=model, data=data, limits=limits.data, contacts=None)
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
        self.assertTrue(np.all(np.isfinite(state_in.body_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_in.body_qd.numpy())))
        self.assertIsInstance(solver._solver_kamino.solver_fd, DVISolver)


    def test_08a_public_solver_heterogeneous_contact_rollout_with_dvi(self):
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        public_basics.make_basics_heterogeneous_builder(builder=builder, ground=True)
        model = builder.finalize(device=self.device, skip_validation_joints=True)

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            use_collision_detector=True,
            collision_detector=kamino_config.CollisionDetectorConfig(
                max_contacts=64 * model.world_count,
                max_contacts_per_world=64,
                max_contacts_per_pair=16,
            ),
        )
        solver = SolverKamino(model, config=config)
        state_in = model.state()
        state_out = model.state()
        control = model.control()

        contact_seen = False
        for _ in range(24):
            solver.step(state_in, state_out, control=control, contacts=None, dt=1.0e-3)
            state_in, state_out = state_out, state_in
            contact_seen = contact_seen or bool(np.any(solver._contacts_kamino.world_active_contacts.numpy() > 0))

        status = solver._solver_kamino.solver_fd.data.status.numpy()
        self.assertTrue(contact_seen)
        self.assertTrue(np.all(np.isfinite(state_in.body_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_in.body_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(status["r_p"])))
        self.assertTrue(np.all(np.isfinite(status["r_d"])))
        self.assertLess(float(np.max(np.abs(state_in.body_qd.numpy()))), 100.0)
        self.assertIsInstance(solver._solver_kamino.solver_fd, DVISolver)
        self.assertTrue(config.sparse_dynamics)
        self.assertTrue(config.sparse_jacobian)
