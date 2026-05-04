# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX hanging-cloth demo.

A cloth grid pinned along its left edge falls under gravity through
the PhoenX position-based cloth iterate. Demonstrates the
end-to-end cloth pipeline:

* Mesh built via :meth:`~newton.ModelBuilder.add_cloth_grid` -- the
  same VBD-compatible API the other Newton cloth solvers consume.
* :meth:`PhoenXWorld.populate_cloth_triangles_from_model` packs the
  triangles into PhoenX's internal constraint-row format and copies
  particle state.
* Per-substep: gravity + position predict (Velocity-level ->
  Position-level access-mode transition); cloth iterate projects
  positions onto constraint manifolds; recover ``velocity =
  (position - position_prev_substep) * inv_dt``
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
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
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

        # PhoenX cloth runs at substeps=4; each substep does one
        # XPBD iterate sweep so the access-mode transitions happen
        # 4x per frame.
        self.sim_substeps = 4
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
        builder = newton.ModelBuilder()
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

        # PhoenX scene: one anchor body (slot 0), no joints, no
        # contacts -- only cloth particles + cloth triangles.
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
            step_layout="single_world",
            device=self.device,
        )
        self.world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))
        self.world.populate_cloth_triangles_from_model(self.model)

        # Newton viewer reads particle state from ``model.state()``;
        # we keep a pair of states and copy PhoenX's particle
        # positions / velocities into ``state_0`` after each step
        # so the renderer always sees the freshest cloth pose.
        self.state_0 = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(6.0, 0.0, 4.0), pitch=-15.0, yaw=180.0)

        self._capture()

    def _capture(self) -> None:
        """Capture the per-frame ``self.world.step + sync`` into a
        CUDA graph so frame replays cost ~10us instead of ~1ms of
        Python launch overhead."""
        if self.device.is_cuda:
            self.world.step(self.frame_dt, contacts=None)  # warm-up
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate_one_frame()
            self.graph = capture.graph
        else:
            self.graph = None

    def _simulate_one_frame(self) -> None:
        """One physics frame. ``contacts=None`` because the demo has
        no rigid bodies / no ground; cloth-cloth and cloth-rigid
        contacts are a follow-up. The cloth iterate runs inside
        :meth:`PhoenXWorld.step` via the dispatcher."""
        self.world.step(self.frame_dt, contacts=None)
        # Mirror PhoenX's particle state into Newton state for
        # rendering.
        wp.copy(self.state_0.particle_q, self.world.particles.position)
        wp.copy(self.state_0.particle_qd, self.world.particles.velocity)

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_one_frame()
        self.sim_time += self.frame_dt

    def test_final(self) -> None:
        """Sanity check after the example finishes: pinned-edge
        particles haven't moved, and the rest of the cloth dropped
        but didn't explode."""
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
