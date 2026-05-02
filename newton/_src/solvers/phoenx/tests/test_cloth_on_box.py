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
from newton._src.solvers.phoenx.examples.example_common import init_phoenx_bodies_kernel
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


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX cloth-on-rigid test requires CUDA")
class TestPhoenxClothWithDynamicBody(unittest.TestCase):
    """Regression: a dynamic rigid body resting on the ground plane
    must stay finite when cloth-cloth contacts are also active in the
    same step.

    Originally found while wiring ``example_cloth_hanging``: the
    cloth-aware singleworld dispatcher's ``relax`` phase was gating
    cloth-triangle cid recognition behind ``wp.static(not is_relax)``,
    so during relax every cloth-triangle cid fell through to
    ``revolute_iterate`` / ``actuated_double_ball_socket_iterate``,
    which interpreted the cloth row's body slots as joint body
    indices and wrote NaN orientation / angular velocity onto whatever
    body slot the cloth row's vertex indices happened to address. The
    failure surfaced as the cube going non-finite on the *second*
    substep of the *first* step, because the first solve's lambdas
    warm-started the second substep through that NaN inertia.
    """

    def setUp(self) -> None:
        self.device = wp.get_device()

    def test_dynamic_cube_on_plane_with_cloth(self) -> None:
        """Cube at rest on ground plane + cloth dropping above must
        stay finite for several frames. Cloth doesn't need to
        actually contact the cube -- the bug fires as soon as RR
        contacts and TT contacts coexist in a relax sweep."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        cube_half = 0.4
        cube_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, cube_half), q=wp.quat_identity()),
        )
        builder.add_shape_box(cube_body, hx=cube_half, hy=cube_half, hz=cube_half)

        # Cloth high above the cube so we get pure RR + TT (no RT)
        # at frame 0; that's the minimal repro of the regression.
        tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5.0e8, 0.3)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 5.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=8,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.05,
            fix_left=True,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            particle_radius=0.04,
        )

        model = builder.finalize(device=self.device)
        num_phoenx_bodies = int(model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        wp.copy(
            bodies.orientation,
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            ),
        )
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=model.body_count,
            inputs=[
                model.body_q,
                model.body_qd,
                model.body_com,
                model.body_inv_mass,
                model.body_inv_inertia,
            ],
            outputs=[
                bodies.position,
                bodies.orientation,
                bodies.velocity,
                bodies.angular_velocity,
                bodies.inverse_mass,
                bodies.inverse_inertia,
                bodies.inverse_inertia_world,
                bodies.motion_type,
                bodies.body_com,
            ],
            device=self.device,
        )

        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=int(model.tri_count),
            device=self.device,
        )
        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(model.particle_count),
            num_cloth_triangles=int(model.tri_count),
            num_worlds=1,
            substeps=4,
            solver_iterations=8,
            step_layout="single_world",
            rigid_contact_max=8192,
            device=self.device,
        )
        world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        world.populate_cloth_triangles_from_model(model)

        pipeline = PhoenxCollisionPipeline(
            model,
            num_cloth_triangles=int(model.tri_count),
            constraints=world.constraints,
            cloth_cid_offset=world.num_joints,
            num_bodies=world.num_bodies,
            particle_q=world.particles.position,
            particle_radius=model.particle_radius,
            cloth_extra_margin=0.05,
            cloth_shape_data_margin=0.0,
            broad_phase="sap",
            contact_matching="sticky",
            rigid_contact_max=8192,
        )
        contacts = pipeline.contacts()

        # +1 phoenx slot offset matching ``init_phoenx_bodies_kernel``.
        shape_body_np = model.shape_body.numpy()
        shape_body_rigid = np.where(shape_body_np < 0, 0, shape_body_np + 1).astype(np.int32)
        shape_body_full = np.concatenate(
            [shape_body_rigid, np.full(int(model.tri_count), -1, dtype=np.int32)]
        )
        shape_body = wp.array(shape_body_full, dtype=wp.int32, device=self.device)

        state = model.state()
        for _ in range(20):
            pipeline.collide(state, contacts)
            world.step(1.0 / 60.0, contacts=contacts, shape_body=shape_body)

        # Cube state must be finite. Slot 1 holds the Newton body 0
        # (the cube); slot 0 is the static anchor.
        cube_pos = bodies.position.numpy()[1]
        cube_orient = bodies.orientation.numpy()[1]
        cube_vel = bodies.velocity.numpy()[1]
        cube_avel = bodies.angular_velocity.numpy()[1]
        self.assertTrue(np.all(np.isfinite(cube_pos)), f"cube pos non-finite: {cube_pos}")
        self.assertTrue(np.all(np.isfinite(cube_orient)), f"cube orient non-finite: {cube_orient}")
        self.assertTrue(np.all(np.isfinite(cube_vel)), f"cube vel non-finite: {cube_vel}")
        self.assertTrue(np.all(np.isfinite(cube_avel)), f"cube angular vel non-finite: {cube_avel}")

        # Cube should still be sitting near its initial spot
        # (-0, 0, 0.4); a small drift from contact resolution is OK.
        self.assertLess(abs(cube_pos[2] - 0.4), 0.05, f"cube fell or jumped: z={cube_pos[2]}")

        # Cloth must stay finite too.
        positions = world.particles.position.numpy()
        self.assertTrue(np.all(np.isfinite(positions)), "cloth particles went non-finite")


if __name__ == "__main__":
    unittest.main()
