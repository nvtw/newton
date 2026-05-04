# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX hanging-cloth demo *with self-collision enabled*.

Sibling of :mod:`example_cloth_hanging`, but the cloth runs through
the cloth-aware collision pipeline so it self-collides as it folds
under gravity.  The cloth is pinned along its left edge and the
remainder swings down through the cloth-vs-cloth (TT) contact path
-- the ``phoenx_cloth_share_vertex_filter`` drops adjacent triangle
pairs in the broadphase, and the surviving pairs go through GJK/MPR
plus the unified ``endpoint_load`` / ``endpoint_apply_impulse``
contact iterate.

A static ground plane is included so the swinging free edge also
exercises the rigid-vs-triangle (RT) contact path against an
infinite half-space.

Cloth thickness ``cloth_margin = 0.01 m`` (1 cm) is used both as
the per-shape ``shape_gap`` for the cloth-triangle suffix in the
collision pipeline and as the surface offset applied at contact
load time.

Run::

    python -m newton.examples phoenx_cloth_hanging_collide
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
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


CLOTH_MARGIN = 0.01  # 1 cm thickness / speculative shell


class Example:
    """A cloth strip pinned on one edge, falling under gravity, with
    self-collision enabled."""

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

        # Cloth-with-self-collision needs a denser substep budget than
        # the contact-free hanging demo so the TT contact path keeps
        # up with the cloth's swing dynamics.
        self.sim_substeps = 8
        self.solver_iterations = 8

        self.dim_x = width
        self.dim_y = height
        self.cell = 0.1
        self.particle_mass = 0.05
        self.cloth_thickness = CLOTH_MARGIN

        self.youngs_modulus = float(youngs_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.tri_ka, self.tri_ke = cloth_lame_from_youngs_poisson_plane_stress(
            self.youngs_modulus, self.poisson_ratio
        )

        # ---- Build the model -----------------------------------------
        # Static ground plane (z = 0) plus the hanging cloth: the cloth
        # exercises both cloth-vs-cloth (TT) self-collision and
        # rigid-vs-triangle (RT) contact against the ground half-space.
        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=1.0)
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
            particle_radius=0.5 * self.cell,
        )
        builder.gravity = -9.81

        self.model = builder.finalize(device=self.device)

        # ---- PhoenXWorld + cloth-aware pipeline -----------------------
        bodies = body_container_zeros(1, device=self.device)
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
            gravity=(0.0, 0.0, -9.81),
            rigid_contact_max=max(4096, 16 * int(self.model.tri_count)),
            step_layout="single_world",
            device=self.device,
        )
        self.world.populate_cloth_triangles_from_model(self.model)

        # Cloth-aware collision pipeline.  ``cloth_margin`` is stamped
        # as the per-shape ``shape_gap`` for the cloth-triangle suffix
        # and is the speculative + thickness offset at contact load
        # time.
        self.collision_pipeline = self.world.setup_cloth_collision_pipeline(
            self.model, cloth_margin=self.cloth_thickness
        )
        self.contacts = self.collision_pipeline.contacts()

        # Only the static ground plane is a rigid shape in this scene,
        # so map ``model.shape_body`` (-1 for static) into PhoenX's
        # convention (0 sentinel slot for static / world-frame shapes).
        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)

        # ---- Newton state for the renderer ---------------------------
        self.state_0 = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(6.0, 0.0, 4.0), pitch=-15.0, yaw=180.0)

        self._capture()

    def _simulate_one_frame(self) -> None:
        """One physics frame.  ``external_aabb_state`` triggers the
        cloth-aware collide path inside :meth:`PhoenXWorld.step`: it
        rebuilds the cloth-triangle AABB suffix, runs the broadphase
        (with the share-vertex filter dropping fan-edge tri pairs),
        and writes contacts before the constraint solve."""
        wp.copy(self.state_0.particle_q, self.world.particles.position)
        wp.copy(self.state_0.particle_qd, self.world.particles.velocity)
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
            external_aabb_state=self.state_0,
        )
        # Mirror PhoenX's particle state into Newton state for rendering.
        wp.copy(self.state_0.particle_q, self.world.particles.position)
        wp.copy(self.state_0.particle_qd, self.world.particles.velocity)

    def _capture(self) -> None:
        """Capture per-frame ``simulate`` into a CUDA graph."""
        if self.device.is_cuda:
            self._simulate_one_frame()  # warm-up
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate_one_frame()
            self.graph = capture.graph
        else:
            self.graph = None

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_one_frame()
        self.sim_time += self.frame_dt

    def test_final(self) -> None:
        """Sanity check after the example finishes: pinned-edge
        particles haven't moved, free corner has dropped, no NaNs.
        We don't require a specific contact count -- depending on the
        substep count and the swing phase the cloth may not yet be
        in a self-collision configuration."""
        positions = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(positions)):
            raise RuntimeError("non-finite particle position in final state")

        pinned_indices = [j * (self.dim_x + 1) for j in range(self.dim_y + 1)]
        free_corner = self.dim_y * (self.dim_x + 1) + self.dim_x

        pinned_drift = np.linalg.norm(
            positions[pinned_indices] - self.model.particle_q.numpy()[pinned_indices],
            axis=1,
        )
        if pinned_drift.max() > 1.0e-3:
            raise RuntimeError(f"pinned particle drifted: max={pinned_drift.max():.4f} m")

        z_initial = self.model.particle_q.numpy()[free_corner, 2]
        z_drop = z_initial - positions[free_corner, 2]
        if z_drop < 0.05:
            raise RuntimeError(f"free corner barely dropped (z_drop={z_drop:.4f} m)")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--width", type=int, default=32, help="Cloth resolution along the long axis")
    parser.add_argument("--height", type=int, default=16, help="Cloth resolution along the short axis")
    parser.add_argument(
        "--youngs-modulus",
        type=float,
        default=5.0e8,
        help="Young's modulus E [Pa].  Higher = less stretchy.",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=0.3,
        help="Poisson ratio nu [-].  Higher = stronger area preservation.",
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
