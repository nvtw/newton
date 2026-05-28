# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX hanging-cloth + cloth-rigid contact demo.

A cloth strip pinned along its left edge falls under gravity. A free
rigid cube drops onto the cloth from above, and a static ground plane
catches everything below. This exercises the full cloth-aware
contact pipeline:

* Cloth-tri PBD elasticity (Jitter2 ``FemTriPBD`` port)
* Cloth-vs-rigid contacts (the cube vs. the cloth, the ground plane
  vs. either side)
* Share-vertex broad-phase filter (no spurious adjacent-tri contacts)
* Cloth-aware contact iterate via the unified endpoint helpers

Run::

    python -m newton._src.solvers.phoenx.examples.example_cloth_hanging
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
from newton._src.solvers.phoenx.picking import Picking, register_with_viewer_gl
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

# Tonge mass splitting (C# PhoenX default). When ``True`` the
# partitioner caps at :data:`MASS_SPLITTING_MAX_COLORED_PARTITIONS`
# colours and any remainder lands in an overflow bucket solved with
# per-(body, partition) copy states. Requires the single-world step
# layout (already used by this example); joint and cloth-triangle
# constraints route through the slot-aware helpers and so coexist
# fine with mass splitting.
ENABLE_MASS_SPLITTING: bool = True
MASS_SPLITTING_MAX_COLORED_PARTITIONS: int = 8

# When ``True`` the cloth's left edge is pinned (the strip hangs).
# When ``False`` the cloth is fully free and falls under gravity.
PIN_CLOTH: bool = False


class Example:
    """Hanging cloth + a falling rigid cube + a static ground plane."""

    def __init__(
        self,
        viewer,
        args=None,
        width: int = 32,
        height: int = 16,
        youngs_modulus: float = 5.0e8,
        poisson_ratio: float = 0.3,
        cube_size: float = 0.4,
        cube_drop_height: float = 5.0,
        cloth_thickness: float = 0.01,
        cloth_gap: float = 0.030,
    ):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.sim_substeps = 8
        self.solver_iterations = 5

        self.dim_x = width
        self.dim_y = height
        self.cell = 0.1
        self.particle_mass = 0.05

        self.youngs_modulus = float(youngs_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.tri_ka, self.tri_ke = cloth_lame_from_youngs_poisson_plane_stress(self.youngs_modulus, self.poisson_ratio)
        self._cube_drop_height = float(cube_drop_height)
        self._cube_half_extent = float(cube_size)
        self._ground_height = 0.0

        builder = newton.ModelBuilder()

        # Static ground plane at z = 0. ``add_ground_plane`` creates an
        # infinite-plane shape attached to the world body; it stops
        # the cloth and cube from falling forever.
        builder.add_ground_plane(height=self._ground_height)

        # Free rigid cube starting above the cloth so it drops onto
        # the cloth's free corner.
        self._cube_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(-1.0, 1.0, cube_drop_height), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(
            self._cube_body,
            hx=self._cube_half_extent,
            hy=self._cube_half_extent,
            hz=self._cube_half_extent,
        )

        # Hanging cloth pinned along the left edge. ``edge_ke`` sets
        # the per-hinge bending stiffness (in N·m/rad) consumed by the
        # PhoenX dihedral-angle bending constraint below; ``edge_kd``
        # is reserved for damping (currently unused by the iterate).
        # When pinned, lay the grid flat (rotated 90° about Z so the
        # left edge runs along Y). When unpinned, stand the grid
        # vertically (rotate 90° about Y) so it falls edge-down rather
        # than face-down — gives a more interesting unpinned drop.
        if PIN_CLOTH:
            cloth_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
        else:
            cloth_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi * 0.5)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 4.0),
            rot=cloth_rot,
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.dim_x,
            dim_y=self.dim_y,
            cell_x=self.cell,
            cell_y=self.cell,
            mass=self.particle_mass,
            fix_left=PIN_CLOTH,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            edge_ke=1.0,
            particle_radius=0.0,
        )

        self.model = builder.finalize(device=self.device)

        # PhoenX bodies layout: slot 0 is the static "world" anchor
        # body, slots 1..N+1 are Newton's dynamic bodies. The cube
        # lives at slot 1.
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        # Identity orientation default for slot 0 (the world anchor).
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
            num_cloth_triangles=int(self.model.tri_count),
            num_cloth_bending=int(self.model.edge_count),
            device=self.device,
        )
        max_thread_blocks = 8 * self.device.sm_count if self.device.is_cuda else None
        self.world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=int(self.model.particle_count),
            num_cloth_triangles=int(self.model.tri_count),
            num_cloth_bending=int(self.model.edge_count),
            num_worlds=1,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            rigid_contact_max=8192,
            step_layout="single_world",
            mass_splitting=ENABLE_MASS_SPLITTING,
            max_colored_partitions=MASS_SPLITTING_MAX_COLORED_PARTITIONS,
            mass_splitting_unrolled=True,
            mass_splitting_batch_size=1,
            max_thread_blocks=max_thread_blocks,
            partitioner_algorithm="greedy",
            device=self.device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        self.world.populate_cloth_triangles_from_model(self.model)
        # Bending hinges: one PhysX-style dihedral-angle constraint per
        # interior edge of the cloth grid. Without these the strip
        # crumples freely; with the small ``edge_ke`` set above the
        # cloth keeps a soft natural curvature as it hangs.
        self.world.populate_cloth_bending_from_model(self.model)
        self.collision_pipeline = self.world.setup_cloth_collision_pipeline(
            self.model,
            cloth_thickness=cloth_thickness,
            cloth_gap=cloth_gap,
            cloth_self_collision=False,
            rigid_contact_max=8192,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(6.0, 0.0, 4.0), pitch=-15.0, yaw=180.0)

        # Picking. PhoenX bodies layout: slot 0 is the static world
        # anchor (skip via zero half-extents), slot 1 is the cube.
        # ``Picking.pick()`` raycasts every body with positive
        # half-extents; the cube + (optionally) the cloth triangles are
        # the only pickable targets in this scene.
        half_extents_np = np.zeros((num_phoenx_bodies, 3), dtype=np.float32)
        # Slot index = newton body index + 1 (slot 0 is the world anchor).
        half_extents_np[self._cube_body + 1] = (cube_size, cube_size, cube_size)
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(
            self.world,
            self._half_extents,
            model=self.model,
            particles=self.world.particles,
        )
        register_with_viewer_gl(self.viewer, self.picking)

        self._capture()

    def _sync_newton_to_phoenx(self) -> None:
        """Push current Newton body state into PhoenX (slot 0 = anchor)."""
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
        """Reverse: PhoenX body state -> Newton state for rendering."""
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
        # Also push particle state into Newton's state for rendering.
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
        # Newton -> PhoenX (cube state may have been touched by the
        # outer host between frames).
        self._sync_newton_to_phoenx()
        # Picking PD force/impulse: rigid path writes ``bodies.force``
        # (consumed by the solver each substep), cloth path writes a
        # velocity impulse directly to the 3 picked particles. Both
        # kernels gate internally on ``pick_*[0] < 0`` so calling
        # unconditionally keeps the graph-capture invariant.
        self.picking.apply_force(dt=self.frame_dt)
        # Run the cloth-aware collision pipeline + solver step.
        self.world.collide(self.state, self.contacts)
        self.world.step(self.frame_dt, contacts=self.contacts)
        # PhoenX -> Newton for rendering.
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

        if PIN_CLOTH:
            pinned_indices = [j * (self.dim_x + 1) for j in range(self.dim_y + 1)]
            pinned_drift = np.linalg.norm(
                positions[pinned_indices] - self.model.particle_q.numpy()[pinned_indices], axis=1
            )
            if pinned_drift.max() > 1.0e-3:
                raise RuntimeError(f"pinned particle drifted: max={pinned_drift.max():.4f} m")

        # Cube fell under gravity (no constraint), so it should have
        # dropped at least a few cm from its initial height.
        cube_q = self.state.body_q.numpy()
        cube_z = float(cube_q[self._cube_body, 2])
        if cube_z > self._cube_drop_height - 0.05:
            raise RuntimeError(f"cube didn't fall (z={cube_z:.4f}, started at {self._cube_drop_height:.4f})")
        if not np.isfinite(cube_z):
            raise RuntimeError(f"cube z went non-finite: {cube_z}")
        floor_z = self._ground_height + self._cube_half_extent
        if cube_z < floor_z - 0.15:
            raise RuntimeError(f"cube fell through ground plane (z={cube_z:.4f}, expected >= {floor_z:.4f})")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--youngs-modulus", type=float, default=5.0e8)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--cube-size", type=float, default=0.4)
    parser.add_argument("--cube-drop-height", type=float, default=5.0)
    parser.add_argument("--cloth-thickness", type=float, default=0.005)
    parser.add_argument("--cloth-gap", type=float, default=0.010)
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        args,
        width=args.width,
        height=args.height,
        youngs_modulus=args.youngs_modulus,
        poisson_ratio=args.poisson_ratio,
        cube_size=args.cube_size,
        cube_drop_height=args.cube_drop_height,
        cloth_thickness=args.cloth_thickness,
        cloth_gap=args.cloth_gap,
    )
    newton.examples.run(example, args)
