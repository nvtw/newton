# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that packed reduced contacts avoid unused per-link response work."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton


class TestReducedResponseRefresh(unittest.TestCase):
    def _run_packed_contact(self, *, force_response=False, force_reconstruction=False):
        """Return contact motion and counts of optional response/pose work."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder()
        body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.4))
        builder.add_articulation([builder.add_joint_free(body)])
        builder.add_ground_plane()
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            joint_mode="reduced",
            step_layout="single_world",
            substeps=2,
            solver_iterations=4,
            sor_boost=1.0,
        )
        bridge = solver._reduced_articulation
        self.assertFalse(bridge.contact_block_system.requires_impulse_response)
        bridge.contact_block_system.requires_impulse_response = force_response
        state = model.state()
        qd = state.joint_qd.numpy()
        qd[0] = 0.2
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        control = model.control()
        contacts = pipeline.contacts()
        history = []
        reconstruct_current = 0
        original_update = bridge.system.update_local_kinematics
        original_factor = bridge.system.factor

        def update(current_state):
            nonlocal reconstruct_current
            if bridge._kinematics_current:
                reconstruct_current += 1
            return original_update(current_state)

        def factor(*args, **kwargs):
            if force_reconstruction:
                kwargs["update_kinematics"] = True
            return original_factor(*args, **kwargs)

        with (
            patch.object(bridge, "_refresh_impulse_response", wraps=bridge._refresh_impulse_response) as refresh,
            patch.object(bridge.system, "factor", side_effect=factor) as factor_calls,
            patch.object(bridge.system, "update_local_kinematics", side_effect=update),
        ):
            for _ in range(12):
                state.clear_forces()
                pipeline.collide(state, contacts)
                self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                solver.step(state, state, control, contacts, 1.0 / 120.0)
                history.append(np.concatenate((state.body_q.numpy().ravel(), state.body_qd.numpy().ravel())))
            self.assertGreater(factor_calls.call_count, 0)
        result = np.asarray(history)
        self.assertTrue(np.isfinite(result).all())
        return result, refresh.call_count, reconstruct_current

    def test_packed_contact_relaxation_skips_unused_response(self):
        """Match contact motion while omitting an unused per-link response cache."""
        result, calls, _ = self._run_packed_contact()
        self.assertEqual(calls, 0, "Packed contacts do not consume per-link impulse responses")
        reference, reference_calls, _ = self._run_packed_contact(force_response=True)
        self.assertGreater(reference_calls, 0)
        np.testing.assert_array_equal(result, reference)

    def test_relaxation_reuses_published_kinematics(self):
        """Preserve contact motion without reconstructing already published frames."""
        result, _, calls = self._run_packed_contact()
        self.assertEqual(calls, 0, "Integration already published current local frames")
        reference, _, reference_calls = self._run_packed_contact(force_reconstruction=True)
        self.assertGreater(reference_calls, 0)
        np.testing.assert_array_equal(result, reference)


if __name__ == "__main__":
    unittest.main()
