"""Systematic root-cause investigation of cloth-on-cube divergence.

Idea: drop ONE rigid box (very heavy) onto a single cloth triangle
suspended by 3 pinned-mass particles.  Trace what each piece of the
solver does, frame by frame.

Hypotheses to test, one at a time:

H1. Contact normal sometimes has a non-vertical component on a flat
    cube top -> origin: GJK/MPR returning skewed normal.
H2. Lambda warm-start over-accumulates across frames so a lifted
    cloth gets bounced back violently.
H3. The contact bias computation in prepare uses ``effective_gap``
    that wrongly includes both shape gaps when one of them is
    primarily a *speculative* shell, not a thickness.

Each test is a small synthetic scenario where only ONE of these
mechanisms can produce the observed effect.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_lame_from_youngs_poisson_plane_stress,
)
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel as _init_phoenx_bodies_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

device = wp.get_device()


def build_world(cloth_dim: int, cell: float, particle_mass: float, cube_he: float,
                cube_density: float, cube_gap: float, cloth_margin: float,
                drop_height: float, substeps: int, iterations: int):
    sheet_w = cloth_dim * cell
    builder = newton.ModelBuilder()
    builder.add_ground_plane(height=0.0)
    cube_body = builder.add_body(xform=wp.transform(p=wp.vec3(0., 0., cube_he), q=wp.quat_identity()))
    builder.add_shape_box(cube_body, hx=cube_he, hy=cube_he, hz=cube_he,
        cfg=newton.ModelBuilder.ShapeConfig(density=cube_density, mu=0.6, gap=cube_gap))
    tri_ka, tri_ke = cloth_lame_from_youngs_poisson_plane_stress(5e7, 0.3)
    builder.add_cloth_grid(pos=wp.vec3(-0.5*sheet_w, -0.5*sheet_w, 2*cube_he+drop_height),
        rot=wp.quat_identity(), vel=wp.vec3(0,0,0), dim_x=cloth_dim, dim_y=cloth_dim,
        cell_x=cell, cell_y=cell, mass=particle_mass, tri_ke=tri_ke, tri_ka=tri_ka,
        particle_radius=0.5*cell)
    builder.gravity = -9.81
    model = builder.finalize(device=device)
    if int(model.body_count) > 0 and int(model.joint_count) > 0:
        tmp = model.state(); newton.eval_fk(model, model.joint_q, model.joint_qd, tmp)
        model.body_q.assign(tmp.body_q); model.body_qd.assign(tmp.body_qd)
    bodies = body_container_zeros(int(model.body_count)+1, device=device)
    wp.copy(bodies.orientation, wp.array(np.tile([0.,0.,0.,1.], (int(model.body_count)+1, 1)).astype(np.float32),
                                         dtype=wp.quatf, device=device))
    wp.launch(_init_phoenx_bodies_kernel, dim=int(model.body_count),
        inputs=[model.body_q, model.body_qd, model.body_com, model.body_inv_mass, model.body_inv_inertia],
        outputs=[bodies.position, bodies.orientation, bodies.velocity, bodies.angular_velocity,
                 bodies.inverse_mass, bodies.inverse_inertia, bodies.inverse_inertia_world,
                 bodies.motion_type, bodies.body_com], device=device)
    constraints = PhoenXWorld.make_constraint_container(num_joints=0, num_cloth_triangles=int(model.tri_count), device=device)
    world = PhoenXWorld(bodies=bodies, constraints=constraints, num_joints=0,
        num_particles=int(model.particle_count), num_cloth_triangles=int(model.tri_count),
        num_worlds=1, substeps=substeps, solver_iterations=iterations, gravity=(0.,0.,-9.81),
        rigid_contact_max=max(4096, 16*int(model.tri_count)), step_layout='single_world', device=device)
    world.populate_cloth_triangles_from_model(model)
    pipeline = world.setup_cloth_collision_pipeline(model, cloth_margin=cloth_margin)
    contacts = pipeline.contacts()
    shape_body_np = model.shape_body.numpy()
    shape_body = wp.array(np.where(shape_body_np<0, 0, shape_body_np+1), dtype=wp.int32, device=device)
    return model, bodies, world, pipeline, contacts, shape_body


def step_one(world, model, bodies, contacts, shape_body, frame_dt=1/60):
    state = model.state()
    wp.copy(state.particle_q, world.particles.position)
    wp.copy(state.particle_qd, world.particles.velocity)
    world.step(dt=frame_dt, contacts=contacts, shape_body=shape_body, external_aabb_state=state)


def hypothesis_1_static_equilibrium():
    """H1: place cloth IN PRECISE rest position above cube.  No drop,
    no initial velocity.  At rest, the only contact force should be
    a vertical normal balancing gravity.  If cloth still drifts
    sideways, contact normal has a horizontal bias.
    """
    print("\n=== H1: static-rest equilibrium check (cloth pre-placed at rest) ===")
    cloth_margin = 0.005
    cube_gap = 0.005
    cloth_dim = 1
    cell = 0.2
    cube_he = 0.65 * cloth_dim * cell  # 0.13
    rest_z = 2 * cube_he + cube_gap + cloth_margin  # geometrical equilibrium
    drop_height = rest_z - 2 * cube_he

    model, bodies, world, pipeline, contacts, shape_body = build_world(
        cloth_dim=cloth_dim, cell=cell, particle_mass=0.1, cube_he=cube_he,
        cube_density=200.0, cube_gap=cube_gap, cloth_margin=cloth_margin,
        drop_height=drop_height, substeps=8, iterations=8)

    initial_xy = world.particles.position.numpy()[:, :2].copy()
    print(f"  cube_top_z={2*cube_he:.4f}  cloth_initial_z={rest_z:.4f}  expected_rest_z={rest_z:.4f}")

    for f in range(60):
        step_one(world, model, bodies, contacts, shape_body)
        v = world.particles.velocity.numpy()
        p = world.particles.position.numpy()
        xy_drift = float(np.linalg.norm(p[:, :2] - initial_xy, axis=1).max())
        if f < 3 or f in (5, 10, 30, 59):
            print(f"  f={f:3d}  z=[{p[:,2].min():+.4f},{p[:,2].max():+.4f}]  "
                  f"v_xy_max={float(np.linalg.norm(v[:,:2],axis=1).max()):+.4f}  "
                  f"v_z_max={float(np.abs(v[:,2]).max()):+.4f}  xy_drift={xy_drift:.4f}")


def hypothesis_2_lambda_carry():
    """H2: drop cloth from a height that DEEPLY penetrates in 1 frame
    so warm-start lambda gets very large.  Then run many frames and
    check if cloth is repeatedly bounced.

    If H2 is the cause, cloth oscillates persistently with no
    decay.  If lambda decays correctly, cloth settles.
    """
    print("\n=== H2: deep-penetration warm-start lambda carry ===")
    model, bodies, world, pipeline, contacts, shape_body = build_world(
        cloth_dim=1, cell=0.2, particle_mass=0.1, cube_he=0.13,
        cube_density=200.0, cube_gap=0.005, cloth_margin=0.005,
        drop_height=0.5, substeps=8, iterations=8)
    initial_xy = world.particles.position.numpy()[:, :2].copy()
    for f in range(120):
        step_one(world, model, bodies, contacts, shape_body)
        v = world.particles.velocity.numpy()
        p = world.particles.position.numpy()
        v_max = float(np.linalg.norm(v, axis=1).max())
        xy_drift = float(np.linalg.norm(p[:,:2] - initial_xy, axis=1).max())
        if f < 5 or f in (10, 20, 40, 60, 90, 119):
            print(f"  f={f:3d} v_max={v_max:7.3f} z=[{p[:,2].min():+.3f},{p[:,2].max():+.3f}] xy_drift={xy_drift:.3f}")


def hypothesis_3_static_cube_no_lambda_carry():
    """H3: same as H2 but the cube is STATIC (density=0, motion=STATIC)
    so its inverse mass = 0.  The cube has the SAME collision shape
    as the dynamic cube, but cannot rebound.  If cloth still
    explodes, the bug is on the cloth side.
    """
    print("\n=== H3: heavy-static cube (inv_mass = 0) ===")
    model, bodies, world, pipeline, contacts, shape_body = build_world(
        cloth_dim=1, cell=0.2, particle_mass=0.1, cube_he=0.13,
        cube_density=0.0, cube_gap=0.005, cloth_margin=0.005,
        drop_height=0.5, substeps=8, iterations=8)
    initial_xy = world.particles.position.numpy()[:, :2].copy()
    for f in range(60):
        step_one(world, model, bodies, contacts, shape_body)
        v = world.particles.velocity.numpy()
        p = world.particles.position.numpy()
        v_max = float(np.linalg.norm(v, axis=1).max())
        xy_drift = float(np.linalg.norm(p[:,:2] - initial_xy, axis=1).max())
        if f < 5 or f in (10, 20, 40, 59):
            print(f"  f={f:3d} v_max={v_max:7.3f} z=[{p[:,2].min():+.3f},{p[:,2].max():+.3f}] xy_drift={xy_drift:.3f}")


if __name__ == "__main__":
    hypothesis_1_static_equilibrium()
    hypothesis_2_lambda_carry()
    hypothesis_3_static_cube_no_lambda_carry()
