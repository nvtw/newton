# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render a sparse ST-FLIP dam break with two-way rigid coupling.

Command: python -m newton.examples mpm_stflip_dam
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverSTFLIP


@wp.kernel(enable_backward=False)
def color_fluid(velocities: wp.array[wp.vec3], colors: wp.array[wp.vec3]):
    """Color particles by speed using a water-blue ramp."""
    particle = wp.tid()
    speed = wp.min(wp.length(velocities[particle]) * 0.35, 1.0)
    colors[particle] = wp.lerp(wp.vec3(0.02, 0.18, 0.55), wp.vec3(0.3, 0.9, 1.0), speed)


class Example:
    """Collapse a water column around a buoyant dynamic sphere."""

    def __init__(self, viewer, options):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_dt = self.frame_dt / options.substeps
        self.sim_substeps = options.substeps
        self.cell_size = options.cell_size
        self.sim_time = 0.0
        self.frame = 0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverSTFLIP.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = 2.0e4
        builder.default_shape_cfg.kd = 100.0
        builder.default_shape_cfg.mu = 0.05

        # Open glass-like tank, 2.4 x 1.2 x 1.4 m.
        wall_color = wp.vec3(0.08, 0.12, 0.18)
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
            hx=1.2,
            hy=0.6,
            hz=0.05,
            color=wall_color,
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(-1.25, 0.0, 0.65), wp.quat_identity()),
            hx=0.05,
            hy=0.6,
            hz=0.7,
            color=wall_color,
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(1.25, 0.0, 0.65), wp.quat_identity()),
            hx=0.05,
            hy=0.6,
            hz=0.7,
            color=wall_color,
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, -0.65, 0.65), wp.quat_identity()),
            hx=1.2,
            hy=0.05,
            hz=0.7,
            color=wall_color,
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.65, 0.65), wp.quat_identity()),
            hx=1.2,
            hy=0.05,
            hz=0.7,
            color=wall_color,
        )

        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.85, 0.0, 0.65), wp.quat_identity()),
            label="fluid-coupled sphere",
        )
        sphere_cfg = builder.default_shape_cfg.copy()
        sphere_cfg.density = 450.0
        builder.add_shape_sphere(body, radius=0.18, cfg=sphere_cfg, color=wp.vec3(1.0, 0.35, 0.08))

        particle_resolution = options.particle_resolution
        if particle_resolution < 1:
            raise ValueError("particle_resolution must be positive")
        spacing = options.cell_size / float(particle_resolution)
        particle_mass = 1000.0 * spacing**3
        builder.add_particle_grid(
            pos=wp.vec3(-1.05, -0.48, 0.08),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=round(64 * particle_resolution / 3),
            dim_y=round(34 * particle_resolution / 3),
            dim_z=round(48 * particle_resolution / 3),
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=particle_mass,
            jitter=0.3 * spacing,
            radius_mean=0.32 * spacing,
        )

        self.model = builder.finalize()
        self.model.soft_contact_ke = 2.0e4
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_mu = 0.02
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.solver = SolverSTFLIP(
            self.model,
            SolverSTFLIP.Config(
                cell_size=options.cell_size,
                tile_size=options.tile_size,
                max_active_tile_count=options.max_active_tiles,
                padding_tiles=1,
                pressure_iterations=options.pressure_iterations,
                particles_per_cell=float(particle_resolution**3),
                transfer_scheme="apic",
                flip_blend=0.97,
                domain_lower=(-1.18, -0.58, 0.02),
                domain_upper=(1.18, 0.58, 1.35),
                max_velocity=15.0,
            ),
        )
        self.colors = wp.empty(self.model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.minimum_occupied_cells = int(0.75 * self.model.particle_count / float(particle_resolution**3))

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(3.2, -4.0, 2.2), pitch=-14.0, yaw=142.0)

        self.graph = None
        if wp.get_device().is_cuda and options.capture:
            self.simulate()
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        if self.frame % 60 == 0:
            self.solver.check_status()
        self.frame += 1
        self.sim_time += self.frame_dt

    def render(self):
        wp.launch(
            color_fluid,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_qd, self.colors],
            device=self.model.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/fluid",
            self.state_0.particle_q,
            self.model.particle_radius,
            self.colors,
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify that the coupled dam break remains bounded and finite."""
        self.solver.check_status()
        newton.examples.test_particle_state(
            self.state_0,
            "ST-FLIP particles remain inside the tank",
            lambda q, qd: (
                q[0] >= -1.181
                and q[0] <= 1.181
                and q[1] >= -0.581
                and q[1] <= 0.581
                and q[2] >= 0.019
                and q[2] <= 1.351
                and wp.length(qd) <= 15.001
            ),
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "ST-FLIP rigid body remains bounded",
            lambda q, qd: (
                wp.abs(wp.transform_get_translation(q)[0]) < 2.0
                and wp.abs(wp.transform_get_translation(q)[1]) < 1.5
                and wp.abs(wp.transform_get_translation(q)[2]) < 5.0
                and wp.length(qd) < 20.0
            ),
            show_body_q=True,
            show_body_qd=True,
        )
        positions = self.state_0.particle_q.numpy()
        occupied_cells = np.unique(np.floor(positions / self.cell_size).astype(np.int32), axis=0).shape[0]
        if occupied_cells < self.minimum_occupied_cells:
            raise AssertionError(
                f"ST-FLIP retained only {occupied_cells} occupied cells, expected at least "
                f"{self.minimum_occupied_cells}"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--cell-size", type=float, default=0.08)
        parser.add_argument("--tile-size", type=int, default=8)
        parser.add_argument("--particle-resolution", type=int, default=3)
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--pressure-iterations", type=int, default=30)
        parser.add_argument("--max-active-tiles", type=int, default=128)
        parser.add_argument("--capture", action="store_true", default=True)
        parser.add_argument("--no-capture", action="store_false", dest="capture")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
