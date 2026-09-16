# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Speculative mesh contacts must track the remaining gap during substeps."""

import unittest

import warp as wp

import newton


class TestReducedSpeculativeContacts(unittest.TestCase):
    def test_closing_mesh_contact_activates_between_collision_updates(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0105), wp.quat_identity()))
        builder.add_shape_mesh(
            body,
            mesh=newton.Mesh.create_box(0.01, 0.01, 0.01),
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0, margin=0.0, gap=0.001),
        )
        builder.add_articulation([builder.add_joint_free(body)])
        builder.add_ground_plane()
        model = builder.finalize(device="cuda:0")
        state = model.state()
        qd = state.joint_qd.numpy()
        qd[2] = -0.1
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            articulation_mode="reduced",
            step_layout="single_world",
            substeps=10,
            solver_iterations=8,
            sor_boost=1.0,
            prepare_refresh_stride=1,
        )
        pipeline.collide(state, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        solver.step(state, state, model.control(), contacts, 0.01)
        self.assertGreaterEqual(float(state.body_q.numpy()[body, 2]), 0.01 - 1.0e-6)


if __name__ == "__main__":
    unittest.main()
