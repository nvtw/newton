# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test that a pyramid of convex hull cubes remains stable.

Builds a single 20-row pyramid (same as example_convex_pile_benchmark)
on a ground plane and verifies the top cube stays near its initial
position after simulation. This catches regressions in convex-convex
contact quality (wrong normals, penetration depths, or missed contacts).
"""

import unittest

import numpy as np
import warp as wp

import newton

# Match example_convex_pile_benchmark.py exactly
CUBE_HALF = 0.4
CUBE_SPACING = 2.1 * CUBE_HALF  # 0.84
PYRAMID_SIZE = 20


class TestConvexPyramidStability(unittest.TestCase):
    def test_pyramid_top_cube_stays_in_place(self):
        """Top cube of a 20-row pyramid should not move significantly."""
        wp.init()

        builder = newton.ModelBuilder()
        builder.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        cube_mesh = newton.Mesh.create_box(
            CUBE_HALF,
            CUBE_HALF,
            CUBE_HALF,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )

        # Build a 20-row pyramid matching the convex pile benchmark layout.
        for level in range(PYRAMID_SIZE):
            num_cubes = PYRAMID_SIZE - level
            row_width = (num_cubes - 1) * CUBE_SPACING
            z_pos = level * CUBE_SPACING + CUBE_HALF

            for i in range(num_cubes):
                x_pos = -row_width / 2.0 + i * CUBE_SPACING

                body = builder.add_body(
                    xform=wp.transform(
                        p=wp.vec3(x_pos, 0.0, z_pos),
                        q=wp.quat_identity(),
                    ),
                )
                builder.add_shape_convex_hull(body, mesh=cube_mesh)

        model = builder.finalize()

        # Record initial position of the top cube (last body added).
        top_body = model.body_count - 1
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        initial_pos = state_0.body_q.numpy()[top_body][:3].copy()

        collision_pipeline = newton.CollisionPipeline(model, broad_phase="sap")
        contacts = collision_pipeline.contacts()
        solver = newton.solvers.SolverXPBD(model, iterations=2, rigid_contact_relaxation=0.8)

        dt = 1.0 / 100.0 / 10  # 100 fps, 10 substeps
        num_steps = 300  # ~0.3 seconds of simulation

        for _ in range(num_steps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

        final_pos = state_0.body_q.numpy()[top_body][:3]

        z_drop = initial_pos[2] - final_pos[2]
        xy_drift = np.sqrt(
            (final_pos[0] - initial_pos[0]) ** 2 + (final_pos[1] - initial_pos[1]) ** 2
        )

        # The top cube should settle slightly but stay in place.
        # z_drop > 0 means it fell, z_drop < 0 means it rose (pushed apart).
        cube_side = 2.0 * CUBE_HALF
        self.assertLess(
            abs(z_drop),
            cube_side,
            f"Top cube moved too much vertically: z_drop={z_drop:.4f} (limit={cube_side})",
        )
        self.assertLess(
            xy_drift,
            cube_side,
            f"Top cube drifted too far horizontally: {xy_drift:.4f}",
        )
        self.assertGreater(
            final_pos[2],
            0.0,
            f"Top cube fell through ground: z={final_pos[2]:.4f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
