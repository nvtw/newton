# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX hanging-cloth demo.

A cloth grid pinned along its left edge falls under gravity through
the PhoenX position-based cloth iterate, draping over a dynamic box
sitting on a ground plane. Demonstrates the end-to-end cloth pipeline
plus cloth-vs-rigid contact resolution against a moving rigid body:

* Mesh built via :meth:`~newton.ModelBuilder.add_cloth_grid` -- the
  same VBD-compatible API the other Newton cloth solvers consume.
* :meth:`PhoenXWorld.populate_cloth_triangles_from_model` packs the
  triangles into PhoenX's internal constraint-row format and copies
  particle state.
* :class:`PhoenxCollisionPipeline` produces cloth-vs-rigid contacts
  every frame (cloth triangles stamped as virtual TRIANGLE shapes,
  shared by Newton's broad/narrow phases with the rigid shapes).
* Per-substep: gravity + position predict (Velocity-level ->
  Position-level access-mode transition); cloth iterate projects
  positions onto constraint manifolds; recover ``velocity =
  (position - position_substep_start) * inv_dt``
  (Position-level -> Velocity-level transition). All three phases
  are dispatched through unified body-or-particle-aware kernels --
  no particle-only launches.

Run::

    python -m newton.examples phoenx_cloth_hanging
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.cloth_collision.pipeline import PhoenxCollisionPipeline
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


class Example:
    """A cloth strip pinned on one edge, falling under gravity."""

    def __init__(
        self,
        viewer,
        args=None,
        width: int = 32,
        height: int = 16,
        youngs_modulus: float = 5.0e8,
        poisson_ratio: float = 0.3,
    ):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # PhoenX cloth runs at substeps=8 / iters=8 in this scene.
        # The cloth-rigid contact iterate is now position-level
        # (apply_triangle_side updates particle position alongside
        # velocity), so a handful of iterations per substep is enough
        # to converge contact penetration to sub-millimetre. Beyond
        # iters=8 the penetration plateaus while runtime keeps
        # climbing linearly; substeps drives both stability and
        # quality more than iterations once XPBD has positional
        # influence over contacts.
        self.sim_substeps = 8
        self.solver_iterations = 8

        self.dim_x = width
        self.dim_y = height
        self.cell = 0.1
        self.particle_mass = 0.05

        # Public material parameters: Young's modulus E [Pa] and
        # Poisson ratio nu [-]. Cloth is a thin sheet so the
        # plane-stress Lamé conversion is what we want -- the 3D
        # ``E*nu / ((1+nu)*(1-2*nu))`` formula blows up at
        # nu = 0.5; plane stress's ``E*nu / (1 - nu**2)`` stays
        # finite for the whole physical range.
        #
        # Defaults: E = 5e8 Pa is a stiff garment / heavy canvas;
        # nu = 0.3 is in the middle of the typical 0.3-0.45 cloth
        # range. The cloth iterate sees ``alpha = 1 / (k * area)``
        # per row, so doubling E roughly halves visible stretch.
        self.youngs_modulus = float(youngs_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.tri_ka, self.tri_ke = cloth_lame_from_youngs_poisson_plane_stress(self.youngs_modulus, self.poisson_ratio)

        # Build the Newton mesh. Cloth pinned along the entire left
        # edge (x == 0) -- those particles' inverse mass goes to
        # zero in the builder, and the PhoenX cloth iterate honours
        # that by leaving them alone in the mass-weighted projection.
        # The 90 deg rotation about world +z lays the cloth flat at
        # ``pos.z``, with the pinned edge running along world +y at
        # x = 0 and the free edge starting at world x = -dim_y * cell.
        # Pinned at z = 2.0 so the 1.6 m swing reaches the floor and
        # the cube sitting on it.
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
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

        # Ground plane + a dynamic box the cloth can drape over and
        # push around. The box sits in the arc the free edge sweeps
        # through as it swings down from x = -1.6 toward the pinned
        # edge at x = 0, so the cloth catches on the box before
        # settling.
        builder.add_ground_plane()

        cube_half = 0.4
        self.cube_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.6, 1.0, cube_half), q=wp.quat_identity()),
        )
        builder.add_shape_box(self.cube_body, hx=cube_half, hy=cube_half, hz=cube_half)

        self.model = builder.finalize(device=self.device)

        # PhoenX bodies: slot 0 is the static world anchor, then one
        # slot per Newton body (the cube). ``init_phoenx_bodies_kernel``
        # writes Newton body i into phoenx slot i+1.
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        wp.copy(
            bodies.orientation,
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            ),
        )
        if self.model.body_count > 0:
            wp.launch(
                init_phoenx_bodies_kernel,
                dim=self.model.body_count,
                inputs=[
                    self.model.body_q,
                    self.model.body_qd,
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

        # Map Newton's ``shape_body`` (``-1`` for world-fixed,
        # ``i`` for shapes attached to Newton body ``i``) into phoenx
        # slot space (``-1 -> 0``, ``i -> i + 1``), then extend with
        # ``-1`` for the cloth-triangle shape slots the pipeline
        # appends after the rigid shapes. The contact ingest reads
        # ``shape_body[shape_idx]`` for every contact pair; without
        # the cloth-tail extension, cloth-shape lookups fault past
        # the array end.
        shape_body_np = self.model.shape_body.numpy()
        shape_body_rigid = np.where(shape_body_np < 0, 0, shape_body_np + 1).astype(np.int32)
        shape_body_full = np.concatenate(
            [shape_body_rigid, np.full(int(self.model.tri_count), -1, dtype=np.int32)]
        )
        self._shape_body = wp.array(shape_body_full, dtype=wp.int32, device=self.device)

        # ``rigid_contact_max`` sizes both the narrow-phase output
        # buffer (in the pipeline below) and phoenx's per-step contact
        # column store; sized to handle a fully-draped grid against
        # the cube + ground with margin. Cloth narrow phase can emit
        # several contacts per triangle pair, so budget ~16 per tri.
        rigid_contact_max = max(8192, int(self.model.tri_count) * 16)

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
            step_layout="single_world",
            rigid_contact_max=rigid_contact_max,
            device=self.device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        self.world.populate_cloth_triangles_from_model(self.model)

        # Cloth-aware collision pipeline: stamps cloth triangles as
        # virtual TRIANGLE shapes appended after the rigid shapes,
        # runs broad/narrow phase across the unified set, and emits
        # contacts the phoenx PGS iterate consumes via the RT subtype.
        # ``cloth_extra_margin`` widens both the broadphase AABB and
        # the narrow-phase speculative range for cloth shapes. A
        # value bigger than ~one cell-spacing turns flat cloth into a
        # contact storm: every diagonal triangle pair (not vertex-
        # sharing, so the broadphase filter keeps them) sits within
        # margin of every neighbour, and the narrow phase emits a TT
        # contact for each. With cell_x = 0.1 this produced ~8000
        # spurious contacts on a 32x16 grid; dropping the margin to
        # the same scale as ``particle_radius`` (0.04) cuts that to
        # essentially zero with no loss in cloth-rigid quality
        # because the rigid side already contributes its own gap.
        self.pipeline = PhoenxCollisionPipeline(
            self.model,
            num_cloth_triangles=int(self.model.tri_count),
            constraints=self.world.constraints,
            cloth_cid_offset=self.world.num_joints,
            num_bodies=self.world.num_bodies,
            particle_q=self.world.particles.position,
            particle_radius=self.model.particle_radius,
            cloth_extra_margin=0.005,
            cloth_shape_data_margin=0.0,
            broad_phase="sap",
            contact_matching="sticky",
            rigid_contact_max=rigid_contact_max,
        )
        self.contacts = self.pipeline.contacts()

        # Newton viewer reads particle + body state from
        # ``model.state()``; we copy PhoenX's particle positions /
        # velocities and rigid body poses into ``state_0`` after
        # each step so the renderer always sees the freshest pose.
        self.state_0 = self.model.state()
        # Seed body_q with the model's initial pose so the cube
        # renders correctly on frame 0 (before the first step's
        # phoenx -> newton sync runs).
        if self.model.body_count > 0:
            wp.copy(self.state_0.body_q, self.model.body_q)
            wp.copy(self.state_0.body_qd, self.model.body_qd)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(5.0, -3.0, 3.0), pitch=-20.0, yaw=200.0)

        self._capture()

    def _capture(self) -> None:
        """Capture the per-frame ``collide + step + sync`` into a
        CUDA graph so frame replays cost ~10us instead of ~1ms of
        Python launch overhead."""
        if self.device.is_cuda:
            # Warm-up: run the full pipeline once outside capture so
            # any first-launch lazy allocations don't get baked into
            # the graph.
            self.pipeline.collide(self.state_0, self.contacts)
            self.world.step(self.frame_dt, contacts=self.contacts, shape_body=self._shape_body)
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate_one_frame()
            self.graph = capture.graph
        else:
            self.graph = None

    def _simulate_one_frame(self) -> None:
        """One physics frame: cloth-aware collide -> phoenx step ->
        mirror state for rendering. The cloth iterate runs inside
        :meth:`PhoenXWorld.step` via the dispatcher."""
        self.pipeline.collide(self.state_0, self.contacts)
        self.world.step(self.frame_dt, contacts=self.contacts, shape_body=self._shape_body)
        wp.copy(self.state_0.particle_q, self.world.particles.position)
        wp.copy(self.state_0.particle_qd, self.world.particles.velocity)
        # Mirror PhoenX's rigid body state (slots 1..N, slot 0 is
        # the static anchor) back into Newton's ``body_q`` /
        # ``body_qd`` so the viewer can draw the cube.
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
                outputs=[self.state_0.body_q, self.state_0.body_qd],
                device=self.device,
            )

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_one_frame()
        self.sim_time += self.frame_dt

    def test_final(self) -> None:
        """Sanity check after the example finishes: pinned-edge
        particles haven't moved, the rest of the cloth dropped, and
        nothing punched through the ground plane."""
        positions = self.state_0.particle_q.numpy()
        # Pinned indices = left edge (x == 0 in grid coordinates).
        pinned_indices = [j * (self.dim_x + 1) for j in range(self.dim_y + 1)]
        # Bottom-right corner is the freest particle.
        free_corner = self.dim_y * (self.dim_x + 1) + self.dim_x

        if not np.all(np.isfinite(positions)):
            raise RuntimeError("non-finite particle position in final state")

        # The pinned corners haven't moved (small numerical floor).
        # We snapshot particle_q at construction; if anything moved
        # by more than 1mm something's wrong.
        # (We compare against the model's initial particle_q because
        # at this point we don't have the pre-step snapshot anymore.)
        pinned_drift = np.linalg.norm(positions[pinned_indices] - self.model.particle_q.numpy()[pinned_indices], axis=1)
        if pinned_drift.max() > 1.0e-3:
            raise RuntimeError(f"pinned particle drifted: max={pinned_drift.max():.4f} m")

        # Free corner has dropped at least a few cm.
        z_initial = self.model.particle_q.numpy()[free_corner, 2]
        z_drop = z_initial - positions[free_corner, 2]
        if z_drop < 0.05:
            raise RuntimeError(f"free corner barely dropped (z_drop={z_drop:.4f} m)")

        # Cloth didn't punch through the ground plane. With the
        # position-level cloth-rigid contact projection (the velocity
        # impulse + matching position update in apply_triangle_side),
        # PGS holds particles within a few millimetres of the plane on
        # this scene -- well within particle_radius (0.04 m).
        z_min = float(positions[:, 2].min())
        if z_min < -0.04:
            raise RuntimeError(f"cloth fell through the ground plane (z_min={z_min:.4f} m)")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--width", type=int, default=32, help="Cloth resolution along the long axis")
    parser.add_argument("--height", type=int, default=16, help="Cloth resolution along the short axis")
    parser.add_argument(
        "--youngs-modulus",
        type=float,
        default=5.0e8,
        help="Young's modulus E [Pa]. Higher = less stretchy.",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=0.3,
        help="Poisson ratio nu [-]. Higher = stronger area preservation.",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        args,
        width=args.width,
        height=args.height,
        youngs_modulus=args.youngs_modulus,
        poisson_ratio=args.poisson_ratio,
    )
    newton.examples.run(example, args)
