# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX cloth-on-cube demo (TT + RT contacts on a static substrate).

A flat cloth grid is dropped onto a cube that itself rests on a
ground plane.  The cube is sized to fully catch the cloth, and the
cloth is unpinned so it conforms to the cube's top face under
gravity and friction.

Run::

    python -m newton._src.solvers.phoenx.examples.example_cloth_on_cube --cloth-dim 12
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
    init_phoenx_bodies_kernel as _init_phoenx_bodies_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    phoenx_to_newton_kernel as _phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


class Example:
    """Single cloth grid drops onto a cube resting on a ground plane.

    The cube's half-extent is sized to match the cloth's diagonal so
    every cloth particle has cube top to land on (no edge-fall-off
    artefacts).  The cloth is unpinned -- it relies entirely on
    contact + friction with the cube to settle.
    """

    def __init__(
        self,
        viewer,
        args=None,
        cloth_dim: int = 12,
        cell: float = 0.1,
    ):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.sim_substeps = 8
        self.solver_iterations = 8

        self.cloth_dim = cloth_dim
        self.cell = cell
        # Heavier per-particle mass keeps the cube : per-cloth-particle
        # ratio in the regime where PGS converges in a few iterations
        # without mass splitting.
        self.particle_mass = 0.1
        self.cloth_thickness = 0.02
        self.youngs_modulus = 5.0e7
        self.poisson_ratio = 0.3
        self.tri_ka, self.tri_ke = cloth_lame_from_youngs_poisson_plane_stress(
            self.youngs_modulus, self.poisson_ratio
        )

        # Cloth side length.
        sheet_w = self.cloth_dim * self.cell

        # Cube must catch the entire cloth even after a small drop and
        # any sideways drift.  Make the cube ~30 % wider than the cloth.
        self.cube_he = 0.65 * sheet_w
        # Cube on top of ground plane: bottom at z = 0, top at z = 2 * cube_he.
        self.cube_center_z = self.cube_he
        self.cube_top_z = 2.0 * self.cube_he

        # ---- Build the model -----------------------------------------
        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=0.0)
        cube_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.cube_center_z), q=wp.quat_identity()),
        )
        # ``gap=0.01`` matches the cloth thickness so the cube's
        # speculative shell doesn't dwarf the cloth's: the builder
        # default ``rigid_gap = 0.1 m`` would over-pump the Baumgarte
        # bias against light cloth particles and produce sustained
        # vertical jitter even after the cloth has nominally settled.
        builder.add_shape_box(
            cube_body,
            hx=self.cube_he,
            hy=self.cube_he,
            hz=self.cube_he,
            cfg=newton.ModelBuilder.ShapeConfig(density=200.0, mu=0.6, gap=0.01),
        )
        self.cube_body = cube_body

        # Cloth above the cube top, centred over the cube.
        sheet_origin_xy = (-0.5 * sheet_w, -0.5 * sheet_w)
        sheet_z = self.cube_top_z + 0.1
        builder.add_cloth_grid(
            pos=wp.vec3(sheet_origin_xy[0], sheet_origin_xy[1], sheet_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.cloth_dim,
            dim_y=self.cloth_dim,
            cell_x=self.cell,
            cell_y=self.cell,
            mass=self.particle_mass,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            particle_radius=0.5 * self.cell,
        )
        builder.gravity = -9.81

        self.model = builder.finalize(device=self.device)

        # FK so body_q reflects the spawn pose.
        if int(self.model.body_count) > 0 and int(self.model.joint_count) > 0:
            tmp = self.model.state()
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, tmp)
            self.model.body_q.assign(tmp.body_q)
            self.model.body_qd.assign(tmp.body_qd)

        # ---- Build PhoenX body container ------------------------------
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
        wp.launch(
            _init_phoenx_bodies_kernel,
            dim=int(self.model.body_count),
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

        # ---- PhoenXWorld + cloth-aware pipeline -----------------------
        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_cloth_triangles=int(self.model.tri_count),
            device=self.device,
        )
        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
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
        self.collision_pipeline = self.world.setup_cloth_collision_pipeline(
            self.model, cloth_margin=self.cloth_thickness
        )
        self.contacts = self.collision_pipeline.contacts()

        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)

        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.0 * sheet_w, -2.0 * sheet_w, sheet_w), pitch=-25.0, yaw=135.0)

        self._capture()

    def _sync_phoenx_to_newton(self) -> None:
        n = self.model.body_count
        if n == 0:
            return
        wp.launch(
            _phoenx_to_newton_kernel,
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

    def _simulate_one_frame(self) -> None:
        self._sync_phoenx_to_newton()
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
            external_aabb_state=self.state,
        )

    def _capture(self) -> None:
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

    def _final_state(self):
        self._sync_phoenx_to_newton()
        wp.copy(self.state.particle_q, self.world.particles.position)
        wp.copy(self.state.particle_qd, self.world.particles.velocity)

    def test_final(self) -> None:
        self._final_state()
        positions = self.state.particle_q.numpy()
        body_q = self.state.body_q.numpy()
        if not np.all(np.isfinite(positions)):
            raise RuntimeError("non-finite particle position in final state")
        if not np.all(np.isfinite(body_q)):
            raise RuntimeError("non-finite body transform in final state")

    def render(self) -> None:
        self._final_state()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--cloth-dim", type=int, default=2, help="Cells per side")
    parser.add_argument("--cell", type=float, default=0.1, help="Cloth cell size [m]")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args=args, cloth_dim=args.cloth_dim, cell=args.cell)
    newton.examples.run(example, args)
