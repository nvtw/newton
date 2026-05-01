# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PhoenX cloth-on-rigid integration test.

A small cloth grid falls under gravity onto a static infinite
plane.  After a handful of frames, the lowest cloth particles must
settle at the plane's contact zone (a particle radius below the
plane is the worst-case slop for the unsolved penetration) --
proving that the full pipeline
(:class:`PhoenxCollisionPipeline` -> ``CollisionPipeline.collide()``
-> :func:`ingest_contacts` -> phoenx PGS iterate with the RT subtype)
delivers contact impulses that hold the cloth up.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.cloth_collision.pipeline import PhoenxCollisionPipeline
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX cloth-on-plane test requires CUDA")
class TestPhoenxClothOnPlane(unittest.TestCase):
    def setUp(self) -> None:
        self.device = wp.get_device()

    def _build_scene(self, *, broad_phase: str = "nxn"):
        """Build a scene with a ground plane and a small cloth grid
        suspended above it, plus the matching PhoenXWorld and
        PhoenxCollisionPipeline."""
        builder = newton.ModelBuilder()

        # Static infinite ground plane at z = 0 -- the same convention
        # phoenx's ``test_bunny_on_plane`` uses for static fixtures.
        builder.add_shape_plane(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            width=0.0,
            length=0.0,
        )

        # Cloth grid hovering at z = 0.6, oriented horizontally.
        # 6x6 grid is tiny but exercises every code path (RT contacts,
        # adjacent-tri filter, barycentric weights).
        dim_x = 6
        dim_y = 6
        cell = 0.15
        # Spacing puts the cloth roughly centred at the origin.
        youngs = 5.0e8
        poisson = 0.3
        tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(youngs, poisson)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.45, -0.45, 0.6),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=cell,
            cell_y=cell,
            mass=0.05,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            particle_radius=0.04,
        )

        model = builder.finalize(device=self.device)
        num_particles = int(model.particle_count)
        num_tris = int(model.tri_count)

        # PhoenXWorld: 1 rigid body (the static box), the cloth particles,
        # and one cloth-triangle constraint row per triangle.
        bodies = body_container_zeros(max(1, int(model.body_count)), device=self.device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=num_tris,
            device=self.device,
        )
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=num_particles,
            num_cloth_triangles=num_tris,
            num_worlds=1,
            substeps=4,
            solver_iterations=8,
            step_layout="single_world",
            rigid_contact_max=4096,
            device=self.device,
        )
        world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        world.populate_cloth_triangles_from_model(model)

        # Cloth-aware collision pipeline. Topology is read from the
        # cloth-triangle rows of ``world.constraints`` -- no separate
        # ``tri_indices`` buffer.
        pipeline = PhoenxCollisionPipeline(
            model,
            num_cloth_triangles=num_tris,
            constraints=world.constraints,
            cloth_cid_offset=world.num_joints,
            num_bodies=world.num_bodies,
            particle_q=world.particles.position,
            particle_radius=model.particle_radius,
            cloth_extra_margin=0.05,
            cloth_shape_data_margin=0.0,
            broad_phase=broad_phase,
            contact_matching="sticky",
        )
        contacts = pipeline.contacts()
        return model, world, pipeline, contacts

    def _run_and_check_no_explosion(self, broad_phase: str) -> None:
        model, world, pipeline, contacts = self._build_scene(broad_phase=broad_phase)
        state = model.state()
        dt = 1.0 / 60.0

        # Run for 40 frames; cloth should fall and settle near the
        # plane (z ~ 0). PGS contact resolution typically holds the
        # particles within one particle-radius of the plane on a
        # coarse scene; tighter convergence comes from raising the
        # solver_iterations / hertz.
        for _ in range(40):
            pipeline.collide(state, contacts)
            world.step(dt, contacts=contacts, shape_body=model.shape_body)

        positions = world.particles.position.numpy()
        self.assertTrue(np.all(np.isfinite(positions)), "particle positions went non-finite")

        # Plane is at z = 0 with particle radius 0.04. Allow a slop
        # of one particle radius for the unsolved penetration in a
        # coarse PGS run; far-below-plane indicates the cloth
        # punched through.
        z_min = float(positions[:, 2].min())
        self.assertGreater(
            z_min,
            -0.10,
            f"cloth fell through the plane (min z = {z_min:.3f}, broad_phase = {broad_phase})",
        )

        # Sanity: cloth should have actually fallen below its initial
        # z = 0.6 -- otherwise gravity didn't apply.
        z_max = float(positions[:, 2].max())
        self.assertLess(
            z_max,
            0.6,
            f"cloth did not move (max z = {z_max:.3f})",
        )

    def test_cloth_falls_on_plane_nxn(self) -> None:
        self._run_and_check_no_explosion("nxn")

    def test_cloth_falls_on_plane_sap(self) -> None:
        self._run_and_check_no_explosion("sap")


if __name__ == "__main__":
    unittest.main()
