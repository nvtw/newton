# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PBF Dam Break
#
# Demonstrates Position-Based Fluids (PBF) integrated into the XPBD solver.
# A column of fluid particles collapses under gravity and interacts with
# rigid bodies and container walls.
#
# Material parameters follow the PhysX PBD default material.
#
# Command: python -m newton.examples pbf_dam_break
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags


@wp.kernel
def _velocity_colors(
    particle_v: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    v_max: float,
    colors: wp.array[wp.vec3],
):
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return
    speed = wp.length(particle_v[tid])
    t = wp.clamp(speed / v_max, 0.0, 1.0)
    r = t
    g = 0.4 + 0.6 * t
    b = 1.0
    colors[tid] = wp.vec3(r, g, b)


class Example:
    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        particle_spacing = args.spacing
        particle_radius = particle_spacing * 0.5
        particle_mass = 1.0

        builder = newton.ModelBuilder()
        builder.default_particle_radius = particle_radius

        # --- Fluid block (dam) ---
        fluid_flags = int(ParticleFlags.ACTIVE | ParticleFlags.FLUID)
        dim_x = args.dim_x
        dim_y = args.dim_y
        dim_z = args.dim_z

        total_particles = dim_x * dim_y * dim_z
        fluid_extent_x = dim_x * particle_spacing
        fluid_extent_y = dim_y * particle_spacing
        fluid_extent_z = dim_z * particle_spacing

        builder.add_particle_grid(
            pos=wp.vec3(
                -fluid_extent_x * 0.85,
                -fluid_extent_y * 0.5,
                particle_spacing,
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=particle_spacing,
            cell_y=particle_spacing,
            cell_z=particle_spacing,
            mass=particle_mass,
            jitter=particle_spacing * 0.1,
            radius_mean=particle_radius,
            flags=fluid_flags,
        )

        print(f"Particle count: {total_particles}")
        print(f"Fluid block: {fluid_extent_x:.2f} x {fluid_extent_y:.2f} x {fluid_extent_z:.2f} m")

        # --- Container walls (invisible static boxes) ---
        wall_thickness = 0.1
        container_hx = fluid_extent_x * 0.9
        container_hy = fluid_extent_y * 0.55
        container_hz = fluid_extent_z * 1.2

        wall_cfg = newton.ModelBuilder.ShapeConfig(mu=0.3, is_visible=False)

        # back wall (-X)
        builder.add_shape_box(
            body=-1, cfg=wall_cfg,
            xform=wp.transform(wp.vec3(-container_hx - wall_thickness, 0.0, container_hz * 0.5), wp.quat_identity()),
            hx=wall_thickness, hy=container_hy + wall_thickness, hz=container_hz * 0.5,
        )
        # front wall (+X)
        builder.add_shape_box(
            body=-1, cfg=wall_cfg,
            xform=wp.transform(wp.vec3(container_hx + wall_thickness, 0.0, container_hz * 0.5), wp.quat_identity()),
            hx=wall_thickness, hy=container_hy + wall_thickness, hz=container_hz * 0.5,
        )
        # left wall (-Y)
        builder.add_shape_box(
            body=-1, cfg=wall_cfg,
            xform=wp.transform(wp.vec3(0.0, -container_hy - wall_thickness, container_hz * 0.5), wp.quat_identity()),
            hx=container_hx + wall_thickness, hy=wall_thickness, hz=container_hz * 0.5,
        )
        # right wall (+Y)
        builder.add_shape_box(
            body=-1, cfg=wall_cfg,
            xform=wp.transform(wp.vec3(0.0, container_hy + wall_thickness, container_hz * 0.5), wp.quat_identity()),
            hx=container_hx + wall_thickness, hy=wall_thickness, hz=container_hz * 0.5,
        )

        # --- Dynamic rigid bodies ---
        box_cfg = newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.3)
        box_size = particle_spacing * 6.0

        b = builder.add_body(xform=wp.transform(wp.vec3(0.2, 0.0, box_size), wp.quat_identity()))
        builder.add_shape_box(body=b, cfg=box_cfg, hx=box_size, hy=box_size, hz=box_size)

        b = builder.add_body(xform=wp.transform(wp.vec3(0.2, 0.0, box_size * 3.5), wp.quat_identity()))
        builder.add_shape_box(body=b, cfg=box_cfg, hx=box_size * 0.7, hy=box_size * 0.7, hz=box_size * 0.7)

        b = builder.add_body(xform=wp.transform(wp.vec3(0.05, 0.2, box_size), wp.quat_identity()))
        builder.add_shape_box(body=b, cfg=box_cfg, hx=box_size * 0.8, hy=box_size * 0.8, hz=box_size * 0.8)

        # Ground
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.3))

        self.model = builder.finalize()
        self.model.set_gravity(args.gravity)

        # PhysX offset derivation (same as createPBDParticleSystem):
        rest_offset = particle_spacing * 0.9
        solid_rest_offset = rest_offset
        fluid_rest_offset = rest_offset * 0.6
        particle_contact_offset = max(solid_rest_offset + 0.001, fluid_rest_offset / 0.6)
        pbf_contact_distance = 2.0 * particle_contact_offset
        fluid_rest_distance = 2.0 * fluid_rest_offset

        print(f"particleContactDistance (h): {pbf_contact_distance:.4f}")
        print(f"fluidRestDistance: {fluid_rest_distance:.4f}")

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=args.iterations,
            pbf_particle_contact_distance=pbf_contact_distance,
            pbf_fluid_rest_distance=fluid_rest_distance,
            pbf_relaxation=args.relaxation,
            pbf_viscosity=args.viscosity,
            pbf_cohesion=args.cohesion,
            pbf_surface_tension=args.surface_tension,
            pbf_vorticity_confinement=args.vorticity,
            pbf_friction=args.pbf_friction,
            pbf_solid_rest_distance=2.0 * solid_rest_offset,
            pbf_cfl_coefficient=args.cfl,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.particle_count = self.model.particle_count
        self.colors = wp.zeros(self.particle_count, dtype=wp.vec3, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        min_z = np.min(q[:, 2])
        max_z = np.max(q[:, 2])
        assert min_z > -0.1, f"Particles fell through ground: min_z={min_z}"
        assert max_z < 5.0, f"Particles exploded: max_z={max_z}"

    def render(self):
        wp.launch(
            kernel=_velocity_colors,
            dim=self.particle_count,
            inputs=[
                self.state_0.particle_qd,
                self.model.particle_flags,
                3.0,
            ],
            outputs=[self.colors],
            device=self.model.device,
        )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "fluid",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.colors,
        )
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0, help="Frames per second")
        parser.add_argument("--spacing", type=float, default=0.025, help="Particle spacing [m]")
        parser.add_argument("--dim-x", type=int, default=46, help="Fluid grid X dimension")
        parser.add_argument("--dim-y", type=int, default=46, help="Fluid grid Y dimension")
        parser.add_argument("--dim-z", type=int, default=46, help="Fluid grid Z dimension")
        # PhysX default PBD material: friction=0.05, viscosity=0.0001,
        # vorticityConfinement=5.0, surfaceTension=0.005, cohesion=0.005
        parser.add_argument("--relaxation", type=float, default=1.0, help="PBF SOR relaxation (PhysX=1.0)")
        parser.add_argument("--viscosity", type=float, default=0.0001, help="PBF viscosity (PhysX default=0.0001)")
        parser.add_argument("--cohesion", type=float, default=0.005, help="PBF cohesion (PhysX default=0.005)")
        parser.add_argument("--surface-tension", type=float, default=0.005, help="PBF surface tension (PhysX default=0.005)")
        parser.add_argument("--vorticity", type=float, default=5.0, help="PBF vorticity confinement (PhysX default=5.0)")
        parser.add_argument("--pbf-friction", type=float, default=0.05, help="PBF particle friction (PhysX default=0.05)")
        parser.add_argument("--cfl", type=float, default=1.0, help="PBF CFL coefficient (PhysX default=1.0)")
        parser.add_argument("--iterations", type=int, default=4, help="XPBD/PBF iterations")
        parser.add_argument("--substeps", type=int, default=8, help="Substeps per frame")
        parser.add_argument(
            "--gravity",
            type=float,
            nargs=3,
            default=[0.0, 0.0, -9.81],
            help="Gravity vector",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
