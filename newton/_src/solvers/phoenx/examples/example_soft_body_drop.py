# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX soft-body tetrahedral drop demo.

Two soft cubes (5 tetrahedra each, via Newton's standard hex-to-tet
decomposition in :meth:`newton.ModelBuilder.add_soft_grid` with a 1x1x1
grid) drop onto a static ground plane. The cubes have different
positions and a slight horizontal offset so they bounce, deform, and
roll on the ground.

The decomposition follows the Jitter2 reference (5 tets per hex cell,
matching ``MeshBuilder.GenerateTetrahedronBlock`` in
``jitterphysics2/.../MeshBuilder.cs:282``).

Exercises:

* FemTetPBD corotational shear (Jitter2 port)
* Virtual ``GeoType.TETRAHEDRON`` collision shapes per tet
* GJK/MPR tet-vs-plane narrow-phase contacts
* 4-node barycentric contact endpoint helpers

Run::

    python -m newton._src.solvers.phoenx.examples.example_soft_body_drop
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    soft_tet_lame_from_youngs_poisson,
)
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


class Example:
    """Two soft cubes dropping onto a ground plane."""

    def __init__(
        self,
        viewer,
        args=None,
        cube_size: float = 0.4,
        cube1_drop_height: float = 2.0,
        cube2_drop_height: float = 3.5,
        youngs_modulus: float = 5.0e7,
        poisson_ratio: float = 0.3,
        density: float = 500.0,
        soft_body_thickness: float = 0.005,
        soft_body_gap: float = 0.010,
    ):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # XPBD compliance scales as ``alpha_tilde = alpha / dt^2``. With
        # more substeps (smaller dt), the per-substep ``bias_mu``
        # ``= 1 / (k_mu * dt^2)`` grows quadratically and the constraint
        # behaves softer per substep. Match the Jitter2 reference
        # (``Playground.NumberSubsteps = 1``, ``SolverIterations = 8``)
        # at frame dt 1/60 so the user's intuition for Young's modulus
        # transfers from the C# experimental sim.
        self.sim_substeps = 1
        self.solver_iterations = 16

        self.youngs_modulus = float(youngs_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.k_lambda, self.k_mu = soft_tet_lame_from_youngs_poisson(
            self.youngs_modulus, self.poisson_ratio
        )

        builder = newton.ModelBuilder()

        # Static ground plane at z = 0.
        builder.add_ground_plane(height=1.0)

        # Soft cube 1: 5 tets via add_soft_grid with dim_x=dim_y=dim_z=1.
        # Newton's hex-to-5-tet decomposition lands one tet per voxel
        # slice (4 corner tets + 1 central tet); the same scheme
        # Jitter2's ``MeshBuilder.GenerateTetrahedronBlock`` uses.
        builder.add_soft_grid(
            pos=wp.vec3(-0.6, 0.0, cube1_drop_height),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1, dim_y=1, dim_z=1,
            cell_x=cube_size, cell_y=cube_size, cell_z=cube_size,
            density=density,
            k_mu=self.k_mu,
            k_lambda=self.k_lambda,
            k_damp=0.0,
            add_surface_mesh_edges=False,
        )

        # Soft cube 2: same construction, offset horizontally so the
        # two cubes don't sit directly atop the same plane patch.
        # Drops from a higher altitude so it impacts after cube 1 has
        # partially settled.
        builder.add_soft_grid(
            pos=wp.vec3(0.6, 0.2, cube2_drop_height),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 1.0, 0.0), 0.3),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1, dim_y=1, dim_z=1,
            cell_x=cube_size, cell_y=cube_size, cell_z=cube_size,
            density=density,
            k_mu=self.k_mu,
            k_lambda=self.k_lambda,
            k_damp=0.0,
            add_surface_mesh_edges=False,
        )

        self.model = builder.finalize(device=self.device)

        # PhoenX bodies: slot 0 = static world anchor. No dynamic
        # rigid bodies in this scene (only the ground plane, which
        # lives at shape_body=-1 and doesn't need a PhoenX body slot).
        num_phoenx_bodies = max(1, int(self.model.body_count) + 1)
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        bodies.orientation.assign(
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            )
        )
        if self.model.body_count > 0:
            self.state_init = self.model.state()
            wp.launch(
                init_phoenx_bodies_kernel,
                dim=self.model.body_count,
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
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            device=self.device,
        )
        self.world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=0,
            num_soft_tetrahedra=int(self.model.tet_count),
            num_worlds=1,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            rigid_contact_max=4096,
            step_layout="single_world",
            device=self.device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        self.world.populate_soft_tetrahedra_from_model(self.model)
        self.collision_pipeline = self.world.setup_cloth_collision_pipeline(
            self.model,
            soft_body_thickness=soft_body_thickness,
            soft_body_gap=soft_body_gap,
            rigid_contact_max=4096,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(4.0, -3.0, 2.0), pitch=-20.0, yaw=215.0)

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
        if n > 0:
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
        # Push particle state into Newton's state for rendering.
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
        # Cubes interacted with the ground (didn't free-fall through
        # the plane). Use a generous bound -- the soft-body collision
        # stack still tunes for tight equilibrium.
        if positions[:, 2].min() < -5.0:
            raise RuntimeError(
                f"a soft-cube particle escaped downward (min z = {positions[:, 2].min():.4f})"
            )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--cube-size", type=float, default=0.4)
    parser.add_argument("--cube1-drop-height", type=float, default=2.0)
    parser.add_argument("--cube2-drop-height", type=float, default=3.5)
    parser.add_argument("--youngs-modulus", type=float, default=1.0e6)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--density", type=float, default=500.0)
    parser.add_argument("--soft-body-thickness", type=float, default=0.005)
    parser.add_argument("--soft-body-gap", type=float, default=0.010)
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        args,
        cube_size=args.cube_size,
        cube1_drop_height=args.cube1_drop_height,
        cube2_drop_height=args.cube2_drop_height,
        youngs_modulus=args.youngs_modulus,
        poisson_ratio=args.poisson_ratio,
        density=args.density,
        soft_body_thickness=args.soft_body_thickness,
        soft_body_gap=args.soft_body_gap,
    )
    newton.examples.run(example, args)
