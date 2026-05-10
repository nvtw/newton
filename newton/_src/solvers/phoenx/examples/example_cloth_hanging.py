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
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

#: Set to ``True`` to draw the contact arrows in the GL viewer
#: (``viewer.log_contacts`` per frame). Off by default because the
#: contact log path reads ``rigid_contact_count`` via ``.numpy()`` every
#: frame, forcing a host sync that breaks the ViewerGL CUDA-OpenGL
#: interop fast path. Toggle this constant or the "Show Contacts"
#: ImGui checkbox to enable; the example calls ``log_contacts`` only
#: when this flag is ``True``, so the checkbox alone is not enough.
SHOW_CONTACTS: bool = True

#: Print PhoenX coloring + contact stats once per frame. Useful to
#: investigate cloth-on-rigid framerate drops (every contact column
#: that shares the rigid body lands in a different colour, so a cube
#: touching N cloth-tris generates ~N colours which translate to ~N
#: kernel launches per PGS iteration).
PRINT_STEP_REPORTS: bool = False

#: Hard caps on the collision-pipeline buffers. Set explicitly here
#: (rather than relying on Newton's heuristics) so the user can tune
#: them deterministically; the GL contact-arrow VBO is sized by
#: ``RIGID_CONTACT_MAX``, so smaller values trade contact headroom
#: for less GPU memory pressure when the "Show Contacts" debug path
#: is on. With the default 32x16 cloth + cube + ground, peak active
#: contacts is well under 4096 (cloth-vs-ground 1024, cloth-vs-cube
#: up to ~1024, plus a handful of cube-ground); 4096 leaves plenty
#: of slack.
RIGID_CONTACT_MAX: int = 4096
#: Broad-phase candidate-pair budget. Worst case is
#: ``num_cloth_tris + num_cloth_tris + 1`` cube-and-ground pairs plus
#: any cloth-cloth pair the share-vertex filter doesn't drop -- a few
#: thousand for the default scene. 8192 is conservative.
SHAPE_PAIRS_MAX: int = 8192


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
        cloth_thickness: float = 0.005,
        cloth_gap: float = 0.010,
    ):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.sim_substeps = 4
        self.solver_iterations = 8

        self.dim_x = width
        self.dim_y = height
        self.cell = 0.1
        self.particle_mass = 0.05

        self.youngs_modulus = float(youngs_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.tri_ka, self.tri_ke = cloth_lame_from_youngs_poisson_plane_stress(
            self.youngs_modulus, self.poisson_ratio
        )
        self._cube_drop_height = float(cube_drop_height)

        builder = newton.ModelBuilder()

        # Static ground plane at z = 0. ``add_ground_plane`` creates an
        # infinite-plane shape attached to the world body; it stops
        # the cloth (and the cube) from falling forever.
        builder.add_ground_plane(height=1.0)

        # Free rigid cube starting above the cloth so it drops onto
        # the cloth's free corner. Mass-1 unit cube; default inertia.
        self._cube_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(-1.0, 1.0, cube_drop_height), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(self._cube_body, hx=cube_size, hy=cube_size, hz=cube_size)

        # Hanging cloth pinned along the left edge.
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 4.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.dim_x,
            dim_y=self.dim_y,
            cell_x=self.cell,
            cell_y=self.cell,
            mass=self.particle_mass,
            fix_left=True,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            particle_radius=0.04,
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
            rigid_contact_max=RIGID_CONTACT_MAX,
            step_layout="single_world",
            device=self.device,
            # Cap the constraint-graph colouring at 12 MIS partitions
            # + one overflow bucket (Jitter2 ``ContactPartitions``
            # convention). When the cube touches the cloth the
            # uncapped coloring spikes to ~192 colours -> ~192
            # iterate kernel launches per PGS iteration; with the
            # cap it stays at 13 colours, and the overflow bucket
            # is processed via mass splitting (per-(body, partition)
            # ``TinyRigidState`` copies + ``1/inv_factor`` impulse
            # scaling). The cloth-aware split iterate variant
            # ensures the cloth-vs-cube contacts are race-free
            # under the cap.
            mass_split_max_partitions=12,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        self.world.populate_cloth_triangles_from_model(self.model)
        self.collision_pipeline = self.world.setup_cloth_collision_pipeline(
            self.model,
            cloth_thickness=cloth_thickness,
            cloth_gap=cloth_gap,
            rigid_contact_max=RIGID_CONTACT_MAX,
            shape_pairs_max=SHAPE_PAIRS_MAX,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(6.0, 0.0, 4.0), pitch=-15.0, yaw=180.0)

        # Pre-warm the contact-arrow GL/CUDA-interop buffer. The first
        # ``viewer.log_contacts`` call lazily allocates a
        # ``RegisteredGLBuffer`` sized to ``rigid_contact_max``; doing
        # that lazily inside ``render()`` AFTER the captured step
        # graph has been replayed N times has been observed to fail
        # with "Failed to allocate ..." on some Warp/CUDA setups.
        # Forcing the alloc before the graph is captured side-steps
        # the issue.
        if SHOW_CONTACTS:
            prev_show = getattr(self.viewer, "show_contacts", False)
            self.viewer.show_contacts = True
            self.viewer.log_contacts(self.contacts, self.state)
            self.viewer.show_contacts = prev_show

        self.frame_index = 0
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
        self.frame_index += 1
        if PRINT_STEP_REPORTS:
            self._print_step_report()

    def _print_step_report(self) -> None:
        """One-line summary of the partitioner / contact state.

        Prints active constraints, contact-column count, and per-colour
        sizes. When the cube touches the cloth, every contact column
        sharing the cube body forces its own colour, so the colour
        count spikes and per-colour sizes drop to 1 -- look for that
        if the framerate suddenly drops.
        """
        report = self.world.step_report()
        slack = (
            f"{report.num_colors / report.max_body_degree:.2f}x"
            if report.max_body_degree > 0
            else "n/a"
        )
        print(
            f"[cloth_hanging] step={self.frame_index} "
            f"contacts={report.num_contact_columns} "
            f"active={report.num_active_constraints} "
            f"colors={report.num_colors} "
            f"max_body_degree={report.max_body_degree} "
            f"colors/lower_bound={slack} "
            f"color_sizes={report.color_sizes}"
        )

    def test_final(self) -> None:
        positions = self.state.particle_q.numpy()
        pinned_indices = [j * (self.dim_x + 1) for j in range(self.dim_y + 1)]

        if not np.all(np.isfinite(positions)):
            raise RuntimeError("non-finite particle position in final state")

        # Pinned particles haven't drifted (small numerical floor).
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
            raise RuntimeError(
                f"cube didn't fall (z={cube_z:.4f}, started at {self._cube_drop_height:.4f})"
            )
        if not np.isfinite(cube_z):
            raise RuntimeError(f"cube z went non-finite: {cube_z}")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        if SHOW_CONTACTS:
            # ``log_contacts`` reads ``rigid_contact_count`` via
            # ``.numpy()`` each frame -- forces a host sync that breaks
            # the ViewerGL CUDA-OpenGL interop fast path. Off by
            # default; flip ``SHOW_CONTACTS`` at the top of this file
            # AND tick the "Show Contacts" checkbox in the GL viewer
            # to draw normal arrows for every active contact.
            self.viewer.log_contacts(self.contacts, self.state)
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
