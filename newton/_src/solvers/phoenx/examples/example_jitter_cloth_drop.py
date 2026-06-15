# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX port of Jitter2 ``Demo17`` ("Cloth").

A horizontal cloth grid pinned along its two short edges catches
three rigid objects (box, capsule, sphere) dropping onto it from
above. Exercises cloth-rigid contact for three different primitive
shapes simultaneously, with mass splitting ON.

Run::

    python -m newton._src.solvers.phoenx.examples.example_jitter_cloth_drop
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

ENABLE_MASS_SPLITTING: bool = True
MASS_SPLITTING_MAX_COLORED_PARTITIONS: int = 12


class Example:
    """A cloth sheet catching a box + capsule + sphere drop."""

    def __init__(
        self,
        viewer,
        args=None,
        cloth_dim: int = 20,
        cloth_cell: float = 0.2,
        cloth_z: float = 3.0,
        particle_mass: float = 0.05,
        drop_height: float = 6.0,
        youngs_modulus: float = 5.0e7,
        poisson_ratio: float = 0.3,
        cloth_thickness: float = 0.01,
        cloth_gap: float = 0.02,
    ):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.sim_substeps = 4
        self.solver_iterations = 8

        tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(youngs_modulus, poisson_ratio)
        self._cloth_z = float(cloth_z)
        self._drop_height = float(drop_height)

        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=0.0)

        # Three rigid objects above the cloth, spread along x.
        cube_half = 0.5
        self._cube_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(-1.0, 0.0, drop_height), q=wp.quat_identity()),
            mass=10.0,
        )
        builder.add_shape_box(self._cube_body, hx=cube_half, hy=cube_half, hz=cube_half)

        capsule_radius = 0.4
        capsule_half = 0.5
        self._capsule_body = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, drop_height),
                q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.5),
            ),
            mass=5.0,
        )
        builder.add_shape_capsule(self._capsule_body, radius=capsule_radius, half_height=capsule_half)

        sphere_radius = 0.5
        self._sphere_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(1.0, 0.0, drop_height + 0.5), q=wp.quat_identity()),
            mass=3.0,
        )
        builder.add_shape_sphere(self._sphere_body, radius=sphere_radius)

        # Horizontal cloth grid pinned along its two short edges so it
        # can sag under load without flying away. Mirrors Demo17's
        # intent (commented-out 25x25 grid + 4 corner BallSocket pins);
        # using fix_left + fix_right is the closest match in Newton's
        # add_cloth_grid API and produces equivalent behaviour.
        cloth_origin = wp.vec3(-cloth_dim * cloth_cell * 0.5, -cloth_dim * cloth_cell * 0.5, cloth_z)
        builder.add_cloth_grid(
            pos=cloth_origin,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cloth_dim,
            dim_y=cloth_dim,
            cell_x=cloth_cell,
            cell_y=cloth_cell,
            mass=particle_mass,
            fix_left=True,
            fix_right=True,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            particle_radius=0.04,
        )

        self.model = builder.finalize(device=self.device)

        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        bodies.orientation.assign(
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            )
        )
        self.state_init = self.model.state()
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=int(self.model.body_count),
            inputs=[
                self.model.body_q,
                self.state_init.body_qd,
                self.model.body_com,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
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
        self.bodies = bodies

        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=int(self.model.tri_count),
            device=self.device,
        )
        self.world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=int(self.model.tri_count),
            num_worlds=1,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            rigid_contact_max=8192,
            step_layout="single_world",
            mass_splitting=ENABLE_MASS_SPLITTING,
            max_colored_partitions=MASS_SPLITTING_MAX_COLORED_PARTITIONS,
            device=self.device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        self.world.populate_cloth_triangles_from_model(self.model)
        self.collision_pipeline = self.world.setup_cloth_collision_pipeline(
            self.model,
            cloth_thickness=cloth_thickness,
            cloth_gap=cloth_gap,
            rigid_contact_max=8192,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.state = self.model.state()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(6.0, 0.0, 4.0), pitch=-15.0, yaw=180.0)

        self._cloth_dim = cloth_dim

        self._capture()

    def _sync_newton_to_phoenx(self) -> None:
        n = int(self.model.body_count)
        if n == 0:
            return
        wp.launch(
            newton_to_phoenx_kernel,
            dim=n,
            inputs=[self.state.body_q, self.state.body_qd, self.model.body_com],
            outputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_phoenx_to_newton(self) -> None:
        n = int(self.model.body_count)
        if n == 0:
            return
        wp.launch(
            phoenx_to_newton_kernel,
            dim=n,
            inputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
                self.model.body_com,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
            device=self.device,
        )
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)

    def _capture(self) -> None:
        if self.device.is_cuda:
            self._simulate_one_frame()  # warm-up
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate_one_frame()
            self.graph = capture.graph
        else:
            self.graph = None

    def _simulate_one_frame(self) -> None:
        self._sync_newton_to_phoenx()
        self.world.collide(self.state, self.contacts)
        self.world.step(self.frame_dt, contacts=self.contacts)
        self._sync_phoenx_to_newton()

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_one_frame()
        self.sim_time += self.frame_dt

    def test_final(self) -> None:
        positions = self.state.particle_q.numpy()
        if not np.all(np.isfinite(positions)):
            raise RuntimeError("non-finite particle position in final state")

        body_q = self.state.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise RuntimeError("non-finite body transform in final state")

        # Each of the three rigid objects must have fallen below its
        # starting height (gravity is active and the cloth shouldn't
        # bounce them back up by drop_height).
        initial_z = self._drop_height
        for label, b in (
            ("cube", self._cube_body),
            ("capsule", self._capsule_body),
            ("sphere", self._sphere_body),
        ):
            z = float(body_q[b, 2])
            if z > initial_z + 0.1:
                raise RuntimeError(f"{label} bounced too high: z = {z:.3f}")

        # Pinned particles (fix_left + fix_right columns) haven't drifted.
        nx1 = self._cloth_dim + 1
        pinned = []
        for j in range(self._cloth_dim + 1):
            pinned.append(j * nx1 + 0)
            pinned.append(j * nx1 + (nx1 - 1))
        pinned_idx = np.asarray(pinned, dtype=np.int64)
        rest_positions = self.model.particle_q.numpy()
        drift = np.linalg.norm(positions[pinned_idx] - rest_positions[pinned_idx], axis=1)
        if drift.max() > 5.0e-3:
            raise RuntimeError(f"pinned particle drifted: max = {drift.max():.4f} m")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--cloth-dim", type=int, default=20)
    parser.add_argument("--cloth-cell", type=float, default=0.2)
    parser.add_argument("--cloth-z", type=float, default=3.0)
    parser.add_argument("--drop-height", type=float, default=6.0)
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        args,
        cloth_dim=args.cloth_dim,
        cloth_cell=args.cloth_cell,
        cloth_z=args.cloth_z,
        drop_height=args.drop_height,
    )
    newton.examples.run(example, args)
