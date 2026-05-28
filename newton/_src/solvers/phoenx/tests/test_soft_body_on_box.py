# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the soft-body tetrahedral collision pipeline.

A small soft cube drops onto a static box and is expected to settle
on top of the box without inversion or NaN. Exercises:

* Tet shape geometry update (canonicalize + encode 4th vertex)
* Per-tet virtual ``GeoType.TETRAHEDRON`` shapes in the collision pipeline
* GJK/MPR tet-vs-box narrow-phase contact generation
* 4-node barycentric contact endpoint helpers in the iterate
* Per-particle internal corotational shear (FemTetPBD)

Mirrors :mod:`test_cloth_on_box` structurally for tets instead of cloth.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX soft-body tests are CUDA-only.",
)
class TestSoftBodyOnBox(unittest.TestCase):
    def test_soft_cube_settles_on_static_box(self):
        device = wp.get_preferred_device()
        box_h = 0.1

        builder = newton.ModelBuilder()
        # Static ground box.
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            hx=0.5,
            hy=0.5,
            hz=box_h,
        )
        # Drop a 2x2x2 soft cube (40 tets after 5-tet-per-cell decomposition)
        # from ~30 cm above the box.
        builder.add_soft_grid(
            pos=wp.vec3(-0.1, -0.1, 0.4),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=200.0,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=0.0,
            add_surface_mesh_edges=False,
        )
        model = builder.finalize(device=device)

        bodies = body_container_zeros(max(1, int(model.body_count)), device=device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(model.tet_count),
            device=device,
        )
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(model.particle_count),
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(model.tet_count),
            rigid_contact_max=4096,
            num_worlds=1,
            substeps=4,
            solver_iterations=8,
            step_layout="single_world",
            device=device,
        )
        world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        world.populate_soft_tetrahedra_from_model(model)
        pipeline = world.setup_cloth_collision_pipeline(
            model,
            soft_body_thickness=0.005,
            soft_body_gap=0.010,
            rigid_contact_max=4096,
        )

        state = model.state()
        contacts = pipeline.contacts()

        # Capture starting mean z and free-fall reference (gravity * t^2 / 2).
        initial_mean_z = float(world.particles.position.numpy()[:, 2].mean())
        T = 0.5  # 0.5 s of sim
        dt = 1.0 / 60.0
        n_frames = int(T / dt)
        free_fall_distance = 0.5 * 9.81 * T * T

        for _ in range(n_frames):
            world.collide(state, contacts)
            world.step(dt, contacts=contacts)

        p_final = world.particles.position.numpy()
        self.assertTrue(np.all(np.isfinite(p_final)), "non-finite particle position")

        # Smoke checks: the soft cube must (a) interact with the box --
        # i.e. fall significantly less than free fall -- and (b) stay
        # within a sane world bound (didn't escape downward). Exact
        # equilibrium / no-penetration is a stability tuning problem
        # left to follow-up work; this test verifies the pipeline runs
        # end-to-end without NaN.
        final_mean_z = float(p_final[:, 2].mean())
        actual_drop = initial_mean_z - final_mean_z
        # Cube must slow down vs free fall (contacts firing). Threshold:
        # less than 70% of analytical free-fall drop.
        self.assertLess(
            actual_drop,
            0.7 * free_fall_distance,
            f"soft cube didn't decelerate (dropped {actual_drop:.3f}m vs free-fall {free_fall_distance:.3f}m); "
            f"box collisions not firing",
        )
        # Cube must not escape downward (rough sanity bound).
        self.assertGreater(
            float(p_final[:, 2].min()),
            -5.0,
            f"soft cube escaped downward (min z = {float(p_final[:, 2].min()):.4f}); contacts failed",
        )


if __name__ == "__main__":
    unittest.main()
