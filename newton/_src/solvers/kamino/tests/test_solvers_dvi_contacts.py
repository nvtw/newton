# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused Kamino DVI Contacts tests."""

from newton._src.solvers.kamino.tests.dvi_test_helpers import *  # noqa: F403


class TestDVIContacts(DVITestCase):
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
        _assert_solver_status_converged(self, container_solver)
        _check_solution_matches_dual_problem(self, problem, container_solver)


    def test_06a_dvi_masked_reset_preserves_unselected_worlds(self):
        builder = builder_utils.make_homogeneous_builder(
            num_worlds=3,
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
        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = _solve_dvi(model, problem)
        lambdas_before = solver.data.solution.lambdas.numpy().copy()
        v_plus_before = solver.data.solution.v_plus.numpy().copy()

        world_mask = wp.array([False, True, False], dtype=wp.bool, device=self.device)
        solver.reset(problem=problem, world_mask=world_mask)

        lambdas_after = extract_problem_vector(
            problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=False
        )
        v_plus_after = extract_problem_vector(
            problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=False
        )
        lambdas_before = extract_problem_vector(problem.delassus, lambdas_before, only_active_dims=False)
        v_plus_before = extract_problem_vector(problem.delassus, v_plus_before, only_active_dims=False)
        np.testing.assert_array_equal(lambdas_after[0], lambdas_before[0])
        np.testing.assert_array_equal(lambdas_after[2], lambdas_before[2])
        np.testing.assert_array_equal(v_plus_after[0], v_plus_before[0])
        np.testing.assert_array_equal(v_plus_after[2], v_plus_before[2])
        np.testing.assert_array_equal(lambdas_after[1], np.zeros_like(lambdas_after[1]))
        np.testing.assert_array_equal(v_plus_after[1], np.zeros_like(v_plus_after[1]))


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

        config = DualProblem.Config(
            constraints=kamino_config.ConstraintStabilizationConfig(
                gamma=0.015,
                delta=0.0,
                contact_recovery_speed=-1.0,
                contact_deep_recovery_gamma=0.08,
                contact_deep_recovery_threshold=0.0025,
            )
        )
        problem_config = wp.array([config.to_struct()], dtype=DualProblemConfigStruct, device=self.device)
        np.testing.assert_allclose(contact_bias(-0.001), [0.0, 0.0, -0.0015], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(contact_bias(-0.005), [0.0, 0.0, -0.02375], rtol=1e-6, atol=1e-6)
