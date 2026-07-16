# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused Kamino DVI Sparse tests."""

from newton._src.solvers.kamino.tests.dvi_test_helpers import *  # noqa: F403


class TestDVISparse(DVITestCase):
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
        problem.delassus.matvec(solver.data.solution.lambdas, full, solver.all_worlds_mask)
        full_np = full.numpy()

        _sparse_delassus_matvec_rows(solver, problem, _SPARSE_DELASSUS_ROWS_JOINTS)
        joint_np = solver.data.state.v_aug.numpy()
        _sparse_delassus_matvec_rows(solver, problem, _SPARSE_DELASSUS_ROWS_UNILATERAL)
        unilateral_np = solver.data.state.v_aug.numpy()

        dim = int(problem.data.dim.numpy()[0])
        njc = int(problem.data.njc.numpy()[0])
        np.testing.assert_allclose(joint_np[:njc], full_np[:njc], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(unilateral_np[njc:dim], full_np[njc:dim], rtol=1e-5, atol=1e-5)
