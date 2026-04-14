#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Profile per-phase timing of the Box3D solver step.

Creates a 100-world scene with 10-box stacks per world (1000 bodies total),
then times each phase of the simulation step separately using
wp.synchronize_device() barriers and time.perf_counter().

Run with: uv run python3 profile_box3d.py
"""

import time
from collections import defaultdict

import numpy as np
import warp as wp

import newton
from newton._src.solvers.box3d.config import Box3DConfig, Softness, compute_softness
from newton._src.solvers.box3d.coloring import coloring_prepare_kernel, prepare_contact_masses_2d
from newton._src.solvers.box3d.convert import (
    convert_bodies_from_box3d,
    convert_bodies_to_box3d,
    convert_contacts_to_box3d,
)
from newton._src.solvers.box3d.kernels_integrate import (
    integrate_positions_2d,
    integrate_velocities_2d,
    update_world_inertia_2d,
)
from newton._src.solvers.box3d.kernels_solve import contact_solve_kernel
from newton._src.solvers.box3d.kernels_store import store_impulses_2d


def build_box_stack(n=10):
    """N-box vertical stack scene (single world template)."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    h = 0.5
    cfg = newton.ModelBuilder.ShapeConfig(density=1.0, mu=0.5)
    for i in range(n):
        shift = -0.01 if i % 2 == 0 else 0.01
        b = builder.add_body(xform=wp.transform(wp.vec3(shift, 0.0, h + 2.0 * h * i)))
        builder.add_shape_box(body=b, hx=h, hy=h, hz=h, cfg=cfg)
    return builder


def build_multi_world_scene(num_worlds=100, boxes_per_stack=10):
    """Build a multi-world scene with box stacks."""
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    sub = build_box_stack(boxes_per_stack)
    for _ in range(num_worlds):
        scene.add_world(sub)
    return scene


def profiled_step(
    solver,
    model,
    state_in,
    state_out,
    contacts,
    dt,
    device,
    timings,
):
    """Run one simulation step, timing each phase individually."""
    cfg = solver._config
    buf = solver._buf
    W = solver._num_worlds

    sub_dt = dt / float(cfg.num_substeps)
    inv_sub_dt = 1.0 / sub_dt if sub_dt > 0.0 else 0.0

    soft = compute_softness(cfg.contact_hertz, cfg.contact_damping_ratio, sub_dt)
    soft_static = compute_softness(
        cfg.contact_hertz * cfg.static_hertz_scale,
        cfg.contact_damping_ratio,
        sub_dt,
    )
    soft_joint = compute_softness(cfg.joint_hertz, cfg.joint_damping_ratio, sub_dt)

    num_joints = model.joint_count
    num_joint_colors = solver._num_joint_colors

    gravity_np = model.gravity.numpy().flatten()[:3]
    gravity_vec = wp.vec3(float(gravity_np[0]), float(gravity_np[1]), float(gravity_np[2]))

    # ── Phase: clear_forces ──
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    state_in.clear_forces()
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    timings["clear_forces"].append(t1 - t0)

    # ── Phase: contacts.clear ──
    t0 = time.perf_counter()
    contacts.clear()
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    timings["contacts_clear"].append(t1 - t0)

    # ── Phase: pipeline.collide ──
    t0 = time.perf_counter()
    solver._pipeline.collide(state_in, contacts)
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    timings["pipeline_collide"].append(t1 - t0)

    # ── Phase: convert_bodies_to_box3d ──
    t0 = time.perf_counter()
    buf.bodies_per_world.zero_()
    if model.body_count > 0:
        wp.launch(
            convert_bodies_to_box3d,
            dim=model.body_count,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                model.body_com,
                model.body_inv_mass,
                model.body_inv_inertia,
                model.body_flags,
                model.body_world,
                model.body_world_start,
                W,
            ],
            outputs=[
                buf.body_pos, buf.body_ori, buf.body_vel, buf.body_ang_vel,
                buf.body_inv_mass, buf.body_inv_inertia, buf.body_com,
                buf.body_delta_pos, buf.body_inv_inertia_body,
                buf.bodies_per_world,
            ],
            device=device,
        )
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    timings["body_convert_to"].append(t1 - t0)

    # ── Phase: convert_contacts_to_box3d ──
    has_contacts = contacts is not None and model.shape_count > 0
    buf.contact_count.zero_()
    t0 = time.perf_counter()
    if has_contacts:
        has_matching = contacts.contact_matching and contacts.rigid_contact_match_index is not None
        match_index = contacts.rigid_contact_match_index if has_matching else buf.contact_count

        wp.launch(
            convert_contacts_to_box3d,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                match_index,
                model.shape_body,
                model.shape_material_mu,
                model.shape_material_restitution,
                model.body_world,
                model.body_world_start,
                state_in.body_q,
                model.body_com,
                model.body_flags,
                buf.prev_normal_impulse,
                buf.prev_friction1_impulse,
                buf.prev_friction2_impulse,
                buf.prev_contact_count,
                cfg.max_contacts_per_world,
            ],
            outputs=[
                buf.raw_body_a, buf.raw_body_b, buf.raw_normal,
                buf.raw_r_a, buf.raw_r_b, buf.raw_base_sep,
                buf.raw_friction, buf.raw_restitution,
                buf.raw_normal_impulse, buf.raw_friction1_impulse,
                buf.raw_friction2_impulse,
                buf.contact_count,
            ],
            device=device,
        )
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    timings["contact_convert"].append(t1 - t0)

    # ── Phase: coloring + prepare masses ──
    t0 = time.perf_counter()
    if has_contacts:
        wp.launch_tiled(
            coloring_prepare_kernel,
            dim=[W],
            inputs=[
                buf.raw_body_a, buf.raw_body_b, buf.raw_normal,
                buf.raw_r_a, buf.raw_r_b, buf.raw_base_sep,
                buf.raw_friction, buf.raw_restitution,
                buf.raw_normal_impulse, buf.raw_friction1_impulse,
                buf.raw_friction2_impulse,
                buf.c_body_a, buf.c_body_b, buf.c_normal,
                buf.c_r_a, buf.c_r_b, buf.c_base_sep,
                buf.c_friction, buf.c_restitution,
                buf.c_normal_impulse, buf.c_friction1_impulse,
                buf.c_friction2_impulse,
                buf.c_total_normal_impulse, buf.c_is_static,
                buf.contact_count, buf.color_offsets,
                buf.color_body_mask, buf.color_to_raw,
                buf.bodies_per_world, buf.body_inv_mass,
                cfg.max_colors,
            ],
            block_dim=cfg.block_dim,
            device=device,
        )
        wp.launch(
            prepare_contact_masses_2d,
            dim=(W, cfg.max_contacts_per_world),
            inputs=[
                buf.c_body_a, buf.c_body_b, buf.c_normal,
                buf.c_r_a, buf.c_r_b,
                buf.body_vel, buf.body_ang_vel,
                buf.body_inv_mass, buf.body_inv_inertia,
                buf.contact_count, cfg.max_contacts_per_world,
            ],
            outputs=[
                buf.c_tangent1, buf.c_tangent2,
                buf.c_normal_mass, buf.c_tangent1_mass, buf.c_tangent2_mass,
                buf.c_rel_vel_normal,
            ],
            device=device,
        )
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    timings["coloring_prepare"].append(t1 - t0)

    # ── Phase: substep loop (velocity integrate + contact solve + position integrate) ──
    max_bodies = cfg.max_bodies_per_world

    vel_int_total = 0.0
    contact_solve_total = 0.0
    pos_int_total = 0.0
    inertia_update_total = 0.0

    for sub in range(cfg.num_substeps):
        is_first = sub == 0
        is_last = sub == cfg.num_substeps - 1

        # Update world-frame inertia
        if sub > 0:
            t0 = time.perf_counter()
            wp.launch(
                update_world_inertia_2d,
                dim=(W, max_bodies),
                inputs=[
                    buf.body_ori, buf.body_inv_mass,
                    buf.body_inv_inertia_body, buf.body_inv_inertia,
                    buf.bodies_per_world,
                ],
                device=device,
            )
            wp.synchronize_device(device)
            t1 = time.perf_counter()
            inertia_update_total += t1 - t0

        # Velocity integration
        t0 = time.perf_counter()
        wp.launch(
            integrate_velocities_2d,
            dim=(W, max_bodies),
            inputs=[
                buf.body_vel, buf.body_ang_vel, buf.body_inv_mass,
                buf.bodies_per_world, gravity_vec,
                cfg.linear_damping, cfg.angular_damping, sub_dt,
            ],
            device=device,
        )
        wp.synchronize_device(device)
        t1 = time.perf_counter()
        vel_int_total += t1 - t0

        # Biased contact solve
        t0 = time.perf_counter()
        for _ in range(cfg.num_velocity_iters):
            wp.launch_tiled(
                contact_solve_kernel,
                dim=[W],
                inputs=[
                    buf.body_vel, buf.body_ang_vel,
                    buf.body_inv_mass, buf.body_inv_inertia,
                    buf.body_delta_pos,
                    buf.c_body_a, buf.c_body_b,
                    buf.c_normal, buf.c_tangent1, buf.c_tangent2,
                    buf.c_r_a, buf.c_r_b, buf.c_base_sep,
                    buf.c_normal_mass, buf.c_tangent1_mass, buf.c_tangent2_mass,
                    buf.c_friction, buf.c_restitution, buf.c_rel_vel_normal,
                    buf.c_normal_impulse, buf.c_friction1_impulse,
                    buf.c_friction2_impulse, buf.c_total_normal_impulse,
                    buf.c_is_static,
                    buf.color_offsets,
                    max_bodies, cfg.max_colors,
                    1,  # use_bias
                    1 if is_first else 0,  # warm_start
                    0,  # no restitution yet
                    inv_sub_dt,
                    soft.bias_rate, soft.mass_scale, soft.impulse_scale,
                    soft_static.bias_rate, soft_static.mass_scale,
                    soft_static.impulse_scale,
                    cfg.contact_speed, cfg.restitution_threshold,
                    buf.body_pos, buf.body_ori,
                    buf.j_body_a, buf.j_body_b, buf.j_type,
                    buf.j_local_anchor_a, buf.j_local_anchor_b,
                    buf.j_hinge_axis_local,
                    buf.j_linear_impulse, buf.j_angular_impulse,
                    buf.j_motor_speed, buf.j_max_motor_torque,
                    buf.j_limit_lower, buf.j_limit_upper,
                    buf.j_limit_enabled,
                    buf.j_lower_impulse, buf.j_upper_impulse,
                    buf.joint_color_offsets,
                    num_joints, num_joint_colors,
                    soft_joint.bias_rate, soft_joint.mass_scale,
                    soft_joint.impulse_scale, sub_dt,
                ],
                block_dim=cfg.block_dim,
                device=device,
            )
        wp.synchronize_device(device)
        t1 = time.perf_counter()
        contact_solve_total += t1 - t0

        # Position integration
        t0 = time.perf_counter()
        wp.launch(
            integrate_positions_2d,
            dim=(W, max_bodies),
            inputs=[
                buf.body_pos, buf.body_ori, buf.body_vel,
                buf.body_ang_vel, buf.body_inv_mass,
                buf.body_delta_pos, buf.bodies_per_world, sub_dt,
            ],
            device=device,
        )
        wp.synchronize_device(device)
        t1 = time.perf_counter()
        pos_int_total += t1 - t0

        # Relaxation contact solve
        t0 = time.perf_counter()
        for _ in range(cfg.num_relaxation_iters):
            wp.launch_tiled(
                contact_solve_kernel,
                dim=[W],
                inputs=[
                    buf.body_vel, buf.body_ang_vel,
                    buf.body_inv_mass, buf.body_inv_inertia,
                    buf.body_delta_pos,
                    buf.c_body_a, buf.c_body_b,
                    buf.c_normal, buf.c_tangent1, buf.c_tangent2,
                    buf.c_r_a, buf.c_r_b, buf.c_base_sep,
                    buf.c_normal_mass, buf.c_tangent1_mass, buf.c_tangent2_mass,
                    buf.c_friction, buf.c_restitution, buf.c_rel_vel_normal,
                    buf.c_normal_impulse, buf.c_friction1_impulse,
                    buf.c_friction2_impulse, buf.c_total_normal_impulse,
                    buf.c_is_static,
                    buf.color_offsets,
                    max_bodies, cfg.max_colors,
                    0,  # no bias
                    0,  # no warm start
                    1 if is_last else 0,  # restitution on last substep
                    inv_sub_dt,
                    soft.bias_rate, soft.mass_scale, soft.impulse_scale,
                    soft_static.bias_rate, soft_static.mass_scale,
                    soft_static.impulse_scale,
                    cfg.contact_speed, cfg.restitution_threshold,
                    buf.body_pos, buf.body_ori,
                    buf.j_body_a, buf.j_body_b, buf.j_type,
                    buf.j_local_anchor_a, buf.j_local_anchor_b,
                    buf.j_hinge_axis_local,
                    buf.j_linear_impulse, buf.j_angular_impulse,
                    buf.j_motor_speed, buf.j_max_motor_torque,
                    buf.j_limit_lower, buf.j_limit_upper,
                    buf.j_limit_enabled,
                    buf.j_lower_impulse, buf.j_upper_impulse,
                    buf.joint_color_offsets,
                    num_joints, num_joint_colors,
                    soft_joint.bias_rate, soft_joint.mass_scale,
                    soft_joint.impulse_scale, sub_dt,
                ],
                block_dim=cfg.block_dim,
                device=device,
            )
        wp.synchronize_device(device)
        t1 = time.perf_counter()
        contact_solve_total += t1 - t0

    timings["inertia_update"].append(inertia_update_total)
    timings["velocity_integrate"].append(vel_int_total)
    timings["contact_solve"].append(contact_solve_total)
    timings["position_integrate"].append(pos_int_total)

    # ── Phase: store impulses ──
    t0 = time.perf_counter()
    wp.launch(
        store_impulses_2d,
        dim=(W, cfg.max_contacts_per_world),
        inputs=[
            buf.c_normal_impulse, buf.c_friction1_impulse,
            buf.c_friction2_impulse, buf.contact_count,
        ],
        outputs=[
            buf.prev_normal_impulse, buf.prev_friction1_impulse,
            buf.prev_friction2_impulse, buf.prev_contact_count,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    timings["impulse_store"].append(t1 - t0)

    # ── Phase: convert bodies back ──
    t0 = time.perf_counter()
    if model.body_count > 0:
        state_out.body_q.assign(state_in.body_q)
        state_out.body_qd.assign(state_in.body_qd)
    if model.body_count > 0:
        wp.launch(
            convert_bodies_from_box3d,
            dim=model.body_count,
            inputs=[
                buf.body_pos, buf.body_ori, buf.body_vel, buf.body_ang_vel,
                buf.body_com,
                model.body_world, model.body_world_start,
                model.body_flags, model.body_com, W,
            ],
            outputs=[state_out.body_q, state_out.body_qd],
            device=device,
        )
    if model.particle_count > 0:
        state_out.particle_q.assign(state_in.particle_q)
        state_out.particle_qd.assign(state_in.particle_qd)
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    timings["body_convert_back"].append(t1 - t0)


def main():
    wp.init()
    device = "cuda:0"

    NUM_WORLDS = 100
    BOXES_PER_STACK = 10
    WARMUP_STEPS = 10
    PROFILE_STEPS = 50

    print("=" * 74)
    print("Box3D Solver Step Profiling")
    print("=" * 74)
    print(f"Worlds: {NUM_WORLDS}, Boxes/stack: {BOXES_PER_STACK}, "
          f"Total bodies: {NUM_WORLDS * BOXES_PER_STACK}")
    print(f"Warmup: {WARMUP_STEPS} steps, Profile: {PROFILE_STEPS} steps")
    print()

    # Build scene
    scene = build_multi_world_scene(NUM_WORLDS, BOXES_PER_STACK)
    model = scene.finalize(device=device)
    state_in = model.state()
    state_out = model.state()

    cfg = Box3DConfig(
        num_substeps=4,
        num_velocity_iters=1,
        num_relaxation_iters=1,
        contact_hertz=30.0,
        enable_graph=False,  # Must be off to profile individual phases
    )
    solver = newton.solvers.SolverBox3D(model, config=cfg)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
    contacts = pipeline.contacts()
    dt = 1.0 / 60.0

    # Store pipeline ref on solver so profiled_step can access it
    solver._pipeline = pipeline

    print("Running warmup steps...")
    for _ in range(WARMUP_STEPS):
        state_in.clear_forces()
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in
    wp.synchronize_device(device)
    print("Warmup complete.\n")

    # Profile
    timings = defaultdict(list)

    print(f"Running {PROFILE_STEPS} profiled steps...")
    # Also time the full step for reference
    full_step_times = []

    for step_i in range(PROFILE_STEPS):
        wp.synchronize_device(device)
        t_full_start = time.perf_counter()

        profiled_step(solver, model, state_in, state_out, contacts, dt, device, timings)

        wp.synchronize_device(device)
        t_full_end = time.perf_counter()
        full_step_times.append(t_full_end - t_full_start)

        state_in, state_out = state_out, state_in

    print("Profiling complete.\n")

    # Report
    print("=" * 74)
    print(f"{'Phase':<30} {'Avg (ms)':>10} {'Std (ms)':>10} {'% of step':>10}")
    print("-" * 74)

    # Compute total solver step from all phases
    phases_order = [
        "clear_forces",
        "contacts_clear",
        "pipeline_collide",
        "body_convert_to",
        "contact_convert",
        "coloring_prepare",
        "inertia_update",
        "velocity_integrate",
        "contact_solve",
        "position_integrate",
        "impulse_store",
        "body_convert_back",
    ]

    phase_labels = {
        "clear_forces": "state.clear_forces()",
        "contacts_clear": "contacts.clear()",
        "pipeline_collide": "pipeline.collide()",
        "body_convert_to": "  body convert to box3d",
        "contact_convert": "  contact convert to box3d",
        "coloring_prepare": "  coloring + prepare masses",
        "inertia_update": "  inertia update (substeps)",
        "velocity_integrate": "  velocity integrate",
        "contact_solve": "  contact solve (bias+relax)",
        "position_integrate": "  position integrate",
        "impulse_store": "  impulse store",
        "body_convert_back": "  body convert back",
    }

    avg_full = np.mean(full_step_times) * 1000.0
    total_accounted = 0.0

    solver_phases = [
        "body_convert_to", "contact_convert", "coloring_prepare",
        "inertia_update", "velocity_integrate", "contact_solve",
        "position_integrate", "impulse_store", "body_convert_back",
    ]

    for phase in phases_order:
        vals = np.array(timings[phase]) * 1000.0  # to ms
        avg = np.mean(vals)
        std = np.std(vals)
        pct = avg / avg_full * 100.0 if avg_full > 0 else 0.0
        total_accounted += avg
        label = phase_labels.get(phase, phase)
        print(f"{label:<30} {avg:>10.3f} {std:>10.3f} {pct:>9.1f}%")

    print("-" * 74)

    # Solver-only subtotal
    solver_total_ms = sum(np.mean(np.array(timings[p]) * 1000.0) for p in solver_phases)
    pct_solver = solver_total_ms / avg_full * 100.0 if avg_full > 0 else 0.0
    print(f"{'solver.step() subtotal':<30} {solver_total_ms:>10.3f} {'':>10} {pct_solver:>9.1f}%")

    # Contact solve breakdown
    contact_solve_ms = np.mean(np.array(timings["contact_solve"]) * 1000.0)
    contact_solve_pct = contact_solve_ms / avg_full * 100.0 if avg_full > 0 else 0.0
    # substeps * (velocity_iters + relaxation_iters) launches total
    n_launches = cfg.num_substeps * (cfg.num_velocity_iters + cfg.num_relaxation_iters)
    per_launch = contact_solve_ms / n_launches if n_launches > 0 else 0.0

    print(f"{'  contact solve per launch':<30} {per_launch:>10.3f} {'':>10} {'':>10}")
    print(f"{'  ({} launches total)':<30}".format(n_launches))

    print("-" * 74)
    print(f"{'Total measured':<30} {total_accounted:>10.3f} {'':>10}")
    print(f"{'Full step (wall)':<30} {avg_full:>10.3f} {'':>10}")
    print(f"{'FPS':<30} {1000.0 / avg_full:>10.1f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
