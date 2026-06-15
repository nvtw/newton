# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for cloth-aware contacts.

A cloth grid drops onto a static box and is expected to settle on
top of the box at ``z = box_top + cloth_thickness``. This exercises
the full cloth-aware pipeline:

* Cloth-triangle shape append + per-step canonicalize (phase 1)
* Contact ingest with shape_endpoints + barycentric weights (phase 2)
* Element emission with up to 6 unified-index nodes (phase 3)
* Endpoint helpers in the iterate (phases 4, 5)

Verifies physical correctness: the cloth comes to rest at the right
height, no NaNs, every particle finite.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX cloth contacts run on CUDA only.",
)
class TestClothOnBox(unittest.TestCase):
    def test_cloth_settles_on_static_box(self):
        device = wp.get_preferred_device()
        cloth_thickness = 0.005  # 5 mm, default in setup_cloth_collision_pipeline
        box_top_z = 0.1
        box_h = 0.1

        builder = newton.ModelBuilder()
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            hx=0.5,
            hy=0.5,
            hz=box_h,
        )
        tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5.0e8, 0.3)
        # Drop a small cloth (8x8 = 64 triangles, 81 particles) from
        # 30 cm above the box.
        builder.add_cloth_grid(
            pos=wp.vec3(-0.4, -0.4, 0.3),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=8,
            dim_y=8,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.05,
            fix_left=False,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            particle_radius=0.04,
        )
        model = builder.finalize(device=device)

        bodies = body_container_zeros(max(1, int(model.body_count)), device=device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=int(model.tri_count),
            device=device,
        )
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(model.particle_count),
            num_cloth_triangles=int(model.tri_count),
            rigid_contact_max=4096,
            num_worlds=1,
            substeps=4,
            solver_iterations=8,
            step_layout="single_world",
            device=device,
        )
        world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        world.populate_cloth_triangles_from_model(model)
        pipeline = world.setup_cloth_collision_pipeline(
            model,
            cloth_thickness=cloth_thickness,
            cloth_gap=0.010,
            rigid_contact_max=4096,
        )

        state = model.state()
        contacts = pipeline.contacts()

        # Run 1 second of sim.
        n_frames = 60
        for _ in range(n_frames):
            world.collide(state, contacts)
            world.step(1.0 / 60.0, contacts=contacts)

        p_final = world.particles.position.numpy()
        self.assertTrue(np.all(np.isfinite(p_final)), "non-finite particle position")

        # Cloth should rest on the box top at z ~= box_top + cloth_thickness.
        # Allow some slack for numerical settling and the speculative gap.
        expected_z = box_top_z + cloth_thickness
        mean_z = float(p_final[:, 2].mean())
        min_z = float(p_final[:, 2].min())
        max_z = float(p_final[:, 2].max())

        self.assertGreater(
            min_z,
            box_top_z - 0.005,
            f"cloth particles penetrated the box (min z = {min_z:.4f}, box top = {box_top_z:.4f})",
        )
        self.assertLess(mean_z, 0.15, f"cloth didn't settle (mean z = {mean_z:.4f}, expected near {expected_z:.4f})")
        self.assertAlmostEqual(
            mean_z,
            expected_z,
            delta=0.02,
            msg=f"cloth settled at z = {mean_z:.4f}, expected near {expected_z:.4f} "
            f"(box_top + thickness); range [{min_z:.4f}, {max_z:.4f}]",
        )


if __name__ == "__main__":
    unittest.main()
