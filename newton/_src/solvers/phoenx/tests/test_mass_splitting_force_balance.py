# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Mass splitting must keep the forces of a resting hub body in balance."""

import unittest

import numpy as np
import warp as wp

import newton

_GRID = 10
_BOX_HALF = 0.03  # [m]
_PLATE_HALF = (0.5, 0.5, 0.05)  # [m]


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX tests run on CUDA with graph capture")
class TestMassSplittingForceBalance(unittest.TestCase):
    def test_ground_carries_plate_and_box_grid(self):
        """A 1 x 1 x 0.1 m plate carries a 10 x 10 grid of boxes on the ground.

        The plate touches every box and the ground, more constraints than the
        four colors hold, so mass splitting gives it several copies. At rest the
        ground must push on the plate with the weight of the plate and all
        boxes, and nothing may sink.
        """
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5)
        plate = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, _PLATE_HALF[2]), wp.quat_identity()))
        builder.add_shape_box(plate, hx=_PLATE_HALF[0], hy=_PLATE_HALF[1], hz=_PLATE_HALF[2], cfg=cfg)
        box_z = 2.0 * _PLATE_HALF[2] + _BOX_HALF
        pitch = 2.0 * _PLATE_HALF[0] / _GRID
        for i in range(_GRID):
            for j in range(_GRID):
                x = -_PLATE_HALF[0] + (i + 0.5) * pitch
                y = -_PLATE_HALF[1] + (j + 0.5) * pitch
                box = builder.add_body(xform=wp.transform(wp.vec3(x, y, box_z), wp.quat_identity()))
                builder.add_shape_box(box, hx=_BOX_HALF, hy=_BOX_HALF, hz=_BOX_HALF, cfg=cfg)
        model = builder.finalize(device="cuda:0")
        model.request_contact_attributes("force")

        state_0, state_1, control = model.state(), model.state(), model.control()
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            substeps=4,
            solver_iterations=6,
            step_layout="single_world",
            mass_splitting=True,
            max_colored_partitions=4,
        )
        dt = 1.0 / 60.0
        with wp.ScopedCapture() as capture:
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            wp.copy(state_0.body_q, state_1.body_q)
            wp.copy(state_0.body_qd, state_1.body_qd)

        shape_body = model.shape_body.numpy()
        ground_force = []
        for frame in range(150):
            wp.capture_launch(capture.graph)
            if frame >= 120:
                solver.update_contacts(contacts)
                count = int(contacts.rigid_contact_count.numpy()[0])
                body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
                body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
                force_z = contacts.force.numpy()[:count, 2]  # on shape0's body
                on_plate = (body0 == plate) & (body1 < 0)
                on_ground = (body1 == plate) & (body0 < 0)
                ground_force.append(float(force_z[on_plate].sum() - force_z[on_ground].sum()))

        # The scene must exercise mass splitting: the plate is split into copies.
        section_end = solver.world._copy_state.section_end.numpy()
        node = plate + 1  # PhoenX body 0 is the static world
        self.assertGreater(int(section_end[node] - section_end[node - 1]), 1)

        weight = float(model.body_mass.numpy().sum()) * 9.81
        mean_force = float(np.mean(ground_force))
        self.assertAlmostEqual(mean_force / weight, 1.0, delta=1.0e-3, msg=f"{mean_force} N vs {weight} N")

        poses = state_0.body_q.numpy()
        self.assertTrue(np.isfinite(poses).all())
        self.assertLess(abs(poses[plate, 2] - _PLATE_HALF[2]), 1.0e-3)
        self.assertLess(float(np.abs(poses[plate + 1 :, 2] - box_z).max()), 2.0e-3)
        self.assertLess(float(np.abs(state_0.body_qd.numpy()).max()), 1.0e-2)


if __name__ == "__main__":
    unittest.main()
