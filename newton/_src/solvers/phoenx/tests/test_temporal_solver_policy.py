# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Public temporal configuration and paired internal motor impulses."""

import unittest

import numpy as np

import newton
from newton._src.solvers.phoenx.tests.test_block_joint_policy import make_model, make_solver
from newton._src.solvers.phoenx.tests.test_contact_coupling import _total_momentum
from newton._src.solvers.phoenx.tests.test_direct_drive import _cuda_with_graph_capture


@unittest.skipUnless(_cuda_with_graph_capture(), "Temporal solver requires CUDA")
class TestTemporalSolverPolicy(unittest.TestCase):
    def test_internal_drive_preserves_momentum(self):
        """Preserve both momenta through public temporal motor steps."""
        model = make_model(40.0)
        solver = make_solver(
            model,
            solver_scheme="tgs",
            mass_splitting=True,
            mass_splitting_color_group_size=4,
            prepare_refresh_stride=1,
        )
        state = model.state()
        control = model.control()
        control.joint_target_q.assign([1.0])
        initial = _total_momentum(model, state)
        for _ in range(10):
            solver.step(state, state, control, None, 0.01)
            np.testing.assert_allclose(_total_momentum(model, state), initial, rtol=0, atol=2e-6)
        self.assertGreater(float(np.linalg.norm(state.body_qd.numpy())), 0.01)
        self.assertEqual(int(solver.world._temporal_contact_state.generation.numpy()[0]), 10)

    def test_property_refresh_rejects_unsupported_physics(self):
        """Reject unsupported edits before mutating cached temporal rows."""
        model = make_model(40.0)
        solver = make_solver(
            model,
            solver_scheme="tgs",
            mass_splitting=True,
            mass_splitting_color_group_size=4,
            prepare_refresh_stride=1,
        )
        direct = solver._direct_equality_system
        original_rows = direct.row_dynamic.ptr
        model.joint_armature.assign([0.1])
        with self.assertRaisesRegex(ValueError, "joint_armature"):
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        self.assertEqual(direct.row_dynamic.ptr, original_rows)
        model.joint_armature.assign([0.0])
        original_limit = model.joint_effort_limit.numpy().copy()
        model.joint_effort_limit.assign([0.2])
        with self.assertRaisesRegex(ValueError, "unbounded"):
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        self.assertEqual(direct.row_dynamic.ptr, original_rows)
        model.joint_effort_limit.assign(original_limit)
        model.joint_target_ke.assign([80.0])
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        self.assertEqual(direct._joint_correction_dt_scale, 1.0)
        state = model.state()
        control = model.control()
        control.joint_target_q.assign([1.0])
        initial = _total_momentum(model, state)
        solver.step(state, state, control, None, 0.01)
        np.testing.assert_allclose(_total_momentum(model, state), initial, rtol=0, atol=2e-6)

    def test_rejects_partition_reuse_before_advancing(self):
        """Require outer-step contact lifetime updates before changing state."""
        model = make_model(40.0)
        solver = make_solver(
            model,
            solver_scheme="tgs",
            mass_splitting=True,
            mass_splitting_color_group_size=4,
            prepare_refresh_stride=1,
        )
        state = model.state()
        before_q = state.body_q.numpy().copy()
        before_qd = state.body_qd.numpy().copy()
        solver.reuse_partition = True
        with self.assertRaisesRegex(ValueError, "contact ingestion"):
            solver.step(state, state, None, None, 0.01)
        np.testing.assert_array_equal(state.body_q.numpy(), before_q)
        np.testing.assert_array_equal(state.body_qd.numpy(), before_qd)

    def test_requires_requested_contact_force(self):
        """Require force allocation before querying temporal contact wrenches."""
        model = make_model(40.0)
        solver = make_solver(
            model,
            solver_scheme="tgs",
            mass_splitting=True,
            mass_splitting_color_group_size=4,
            prepare_refresh_stride=1,
        )
        contacts = model.contacts()
        with self.assertRaisesRegex(ValueError, "not allocated"):
            solver.update_contacts(contacts)

    def test_empty_contact_force_output(self):
        """Return an empty wrench buffer for a mechanism without shapes."""
        model = make_model(40.0)
        model.request_contact_attributes("force")
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = make_solver(
            model,
            solver_scheme="tgs",
            mass_splitting=True,
            mass_splitting_color_group_size=4,
            prepare_refresh_stride=1,
        )
        state = model.state()
        solver.step(state, state, None, contacts, 0.01)
        solver.update_contacts(contacts, state)
        np.testing.assert_array_equal(contacts.force.numpy(), 0)

    def test_rejects_incompatible_configuration(self):
        """Reject incompatible temporal solver settings and armature."""
        model = make_model(40.0)
        options = {
            "solver_scheme": "tgs",
            "mass_splitting": True,
            "mass_splitting_color_group_size": 4,
            "prepare_refresh_stride": 1,
        }
        for override in (
            {"mass_splitting": False},
            {"prepare_refresh_stride": 2},
            {"contact_chunk_size": 6},
            {"velocity_readout": "finite_difference"},
            {"velocity_readout": "substep_average"},
            {"solver_scheme": "unknown"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                make_solver(model, **(options | override))
        model.joint_armature.assign([0.1])
        with self.assertRaisesRegex(ValueError, "joint_armature"):
            make_solver(model, **options)


if __name__ == "__main__":
    unittest.main()
