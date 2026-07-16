# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused Kamino DVI DrLegs tests."""

from newton._src.solvers.kamino.tests.dvi_test_helpers import *  # noqa: F403


class TestDVIDrLegs(DVITestCase):
    def test_08b_dr_legs_contact_capacity_scales_with_world_count(self):
        if not self.device.is_cuda:
            self.skipTest("Dr Legs multi-world capacity regression uses the CUDA graph path")

        from types import SimpleNamespace  # noqa: PLC0415

        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        world_count = 3
        args = SimpleNamespace(
            world_count=world_count,
            use_kamino_contacts=False,
            dynamics_solver="dvi",
            dvi_contact_block_preconditioner=False,
            dvi_contact_jacobi_omega=0.45,
            dvi_contact_jacobi_relaxation=0.9,
        )
        example = Example(ViewerNull(num_frames=1), args)

        expected_capacity = 72 * world_count
        self.assertEqual(example.model.rigid_contact_max, expected_capacity)
        self.assertEqual(example.contacts.rigid_contact_max, expected_capacity)
        self.assertEqual(example.collision_pipeline.rigid_contact_max, expected_capacity)


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
            dvi_contact_jacobi_omega=0.45,
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
            dvi_contact_jacobi_omega=0.45,
            dvi_contact_jacobi_relaxation=0.9,
        )
        example = Example(ViewerNull(num_frames=1), args)

        q_tip = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.pi * 0.5))
        example.base_q.assign([wp.transformf((0.0, 0.0, 0.25), q_tip)])
        reset_config = SolverKamino.ResetConfig(base_pose=SolverKamino.ResetConfig.FromBaseQ(example.base_q))
        example.solver.reset(state=example.state_0, config=reset_config)
        example.solver.reset(state=example.state_1, config=reset_config)
        example.capture()

        base_start = example.state_0.body_q.numpy()[0, :3].copy()
        contact_seen = False
        post_settle_penetration = []
        post_settle_xy = []
        for step_idx in range(400):
            example.step()
            contact_seen = contact_seen or int(example.contacts.rigid_contact_count.numpy()[0]) > 0
            contacts_kamino = example.solver._contacts_kamino
            contact_count = int(contacts_kamino.world_active_contacts.numpy()[0])
            if step_idx >= 40 and contact_count > 0:
                gaps = contacts_kamino.gapfunc.numpy()[:contact_count, 3]
                post_settle_penetration.append(float(max(0.0, -np.min(gaps))))
            if step_idx >= 200:
                post_settle_xy.append(example.state_0.body_q.numpy()[0, :2].copy())

        body_q = example.state_0.body_q.numpy()
        body_qd = example.state_0.body_qd.numpy()
        base_delta_xy = body_q[0, :2] - base_start[:2]

        self.assertTrue(contact_seen)
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(body_qd)))
        self.assertLess(float(np.linalg.norm(base_delta_xy)), 0.008)
        self.assertGreater(len(post_settle_penetration), 0)
        self.assertLess(float(np.percentile(post_settle_penetration, 95)), 0.0035)
        self.assertLess(float(np.linalg.norm(post_settle_xy[-1] - post_settle_xy[0])), 5.0e-4)


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
            dvi_contact_jacobi_omega=0.45,
            dvi_contact_jacobi_relaxation=0.9,
        )
        example = Example(ViewerNull(num_frames=1), args)

        base_z = []
        for _ in range(180):
            example.step()
            base_z.append(float(example.state_0.body_q.numpy()[0, 2]))

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
        z = np.array(base_z[60:], dtype=np.float64)
        x = np.arange(z.size, dtype=np.float64)
        residual = z - np.polyval(np.polyfit(x, z, 1), x)
        self.assertLess(float(np.max(residual) - np.min(residual)), 0.001)
