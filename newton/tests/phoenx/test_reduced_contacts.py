# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact response in generalized coordinates."""

import unittest

import warp as wp

import newton


class TestReducedContacts(unittest.TestCase):
    def test_wide_cpu_contact_page_supports_floating_body(self):
        self._check_wide_contact_page("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "requires CUDA")
    def test_wide_cuda_contact_page_supports_floating_body(self):
        self._check_wide_contact_page("cuda")

    def _check_wide_contact_page(self, device):
        """Twelve floor contacts produce 36 response rows, including the vertical translation."""
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            link = builder.add_link(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()))
            for x in (-0.3, -0.1, 0.1, 0.3):
                for y in (-0.2, 0.0, 0.2):
                    builder.add_shape_sphere(link, radius=0.05, xform=wp.transform((x, y, 0.0), wp.quat_identity()))
            builder.add_articulation([builder.add_joint_free(link)])
            builder.add_ground_plane()
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
            solver = newton.solvers.SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                joint_mode="reduced",
                reduced_backend="aba",
                substeps=1,
                solver_iterations=8,
            )
            state, contacts, control = model.state(), pipeline.contacts(), model.control()
            for _ in range(30):
                pipeline.collide(state, contacts)
                solver.step(state, state, control, contacts, 1.0 / 120.0)
            self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 12)
            self.assertAlmostEqual(float(state.body_q.numpy()[link, 2]), 0.05, delta=0.001)
            self.assertLess(abs(float(state.body_qd.numpy()[link, 2])), 0.01)


if __name__ == "__main__":
    unittest.main()
