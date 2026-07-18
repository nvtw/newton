# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproducible throughput benchmark for PhoenX mini checkpoints."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

import newton

from .solver import MiniSolver, MiniSolverConfig


def _make_stack_model(world_count: int, bodies_per_world: int, device: str):
    template = newton.ModelBuilder(up_axis=newton.Axis.Z)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.6)
    for body_index in range(bodies_per_world):
        body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.48 + 0.96 * body_index), wp.quat_identity()))
        template.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, world_count)
    builder.add_ground_plane(cfg=shape_cfg)
    return builder.finalize(device=device)


def _make_robot_model(world_count: int, bodies_per_world: int, device: str):
    if bodies_per_world < 2:
        raise ValueError("robot scene requires at least two bodies per world")
    template = newton.ModelBuilder(up_axis=newton.Axis.Z)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, collision_group=-2)
    ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8)
    half_length = 0.15
    links = []
    for _ in range(bodies_per_world):
        link = template.add_link()
        template.add_shape_box(link, hx=0.08, hy=0.08, hz=half_length, cfg=shape_cfg)
        links.append(link)

    joints = [
        template.add_joint_revolute(
            parent=-1,
            child=links[0],
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0 * half_length * bodies_per_world - 0.02)),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, half_length)),
        )
    ]
    for link_index in range(1, bodies_per_world):
        joints.append(
            template.add_joint_revolute(
                parent=links[link_index - 1],
                child=links[link_index],
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -half_length)),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, half_length)),
            )
        )
    template.add_articulation(joints, label="mini_robot_chain")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, world_count)
    builder.add_ground_plane(cfg=ground_cfg)
    return builder.finalize(device=device)


def _make_model(scene: str, world_count: int, bodies_per_world: int, device: str):
    if scene == "robot":
        return _make_robot_model(world_count, bodies_per_world, device)
    return _make_stack_model(world_count, bodies_per_world, device)


def _run(args: argparse.Namespace) -> dict[str, float | int | str | None]:
    device = wp.get_device(args.device)
    model = _make_model(args.scene, args.worlds, args.bodies_per_world, args.device)
    contact_matching = args.contact_matching
    if contact_matching == "auto":
        contact_matching = "sticky" if args.solver == "phoenx" else "disabled"
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase=args.broad_phase,
        rigid_contact_max=args.worlds * args.max_contacts_per_world,
        contact_matching=contact_matching,
    )
    contacts = pipeline.contacts()
    state_0 = model.state()
    state_1 = model.state()
    if args.scene == "robot":
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)
    control = model.control()
    if args.solver == "mini":
        solver = MiniSolver(
            model,
            MiniSolverConfig(
                substeps=args.substeps,
                iterations=args.iterations,
                block_dim=args.block_dim,
                max_colors=args.max_colors,
                max_constraints_per_world=args.max_constraints_per_world,
                max_constraints_per_color=args.max_constraints_per_color,
                shared_body_cache=args.shared_body_cache,
                solve_layout=args.solve_layout,
            ),
        )
    else:
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=args.substeps,
            solver_iterations=args.iterations,
            velocity_iterations=0,
            contact_friction_model="point",
            step_layout="multi_world",
            threads_per_world=args.phoenx_threads_per_world,
            articulation_mode="maximal",
        )

    def step() -> None:
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, args.dt)
        wp.copy(state_0.body_q, state_1.body_q)
        wp.copy(state_0.body_qd, state_1.body_qd)

    for _ in range(args.settle_steps):
        step()
    with wp.ScopedCapture(device=device) as capture:
        step()
    for _ in range(args.warmup):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(args.replays):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - start

    poses = state_0.body_q.numpy()
    velocities = state_0.body_qd.numpy()
    if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
        raise RuntimeError("mini benchmark produced non-finite state")
    stats = solver.stats() if args.solver == "mini" else None
    contacts_per_step = int(contacts.rigid_contact_count.numpy()[0])
    joint_types = model.joint_type.numpy() if model.joint_count else np.empty(0, dtype=np.int32)
    revolute_constraints = int(np.count_nonzero(joint_types == int(newton.JointType.REVOLUTE)))
    world_steps = args.worlds * args.replays
    body_steps = model.body_count * args.replays
    iteration_scale = args.iterations * args.substeps * args.replays
    constraint_iterations = contacts_per_step * iteration_scale
    constraint_iteration_rate = constraint_iterations / elapsed
    grouped_constraint_rate = (contacts_per_step + revolute_constraints) * iteration_scale / elapsed
    scalar_row_rate = (3 * contacts_per_step + 5 * revolute_constraints) * iteration_scale / elapsed
    packed_contact_path = args.solver == "mini" and solver._packed_contacts
    packed_mixed_path = args.solver == "mini" and solver._packed_mixed
    if packed_contact_path:
        logical_bytes = contacts_per_step * 352 * iteration_scale
        estimated_flops = contacts_per_step * 450 * iteration_scale
        roofline_model = "C2 packed contact algorithmic lower bound; no GPU counters"
    elif packed_mixed_path:
        logical_bytes = (contacts_per_step * 352 + revolute_constraints * 400) * iteration_scale
        estimated_flops = (contacts_per_step * 450 + revolute_constraints * 600) * iteration_scale
        roofline_model = "C4 packed contact/revolute algorithmic lower bound; no GPU counters"
    elif args.solver == "mini":
        logical_bytes = (contacts_per_step * 348 + revolute_constraints * 320) * iteration_scale
        estimated_flops = (contacts_per_step * 1100 + revolute_constraints * 1500) * iteration_scale
        roofline_model = "C1 contact/revolute algorithmic lower bound; no GPU counters"
    else:
        # Use the same prepared-row lower bound as mini C4 so full/mini
        # throughput is compared on identical useful physics work. This omits
        # full-PhoenX matching, ingest, sorting, and scheduling traffic.
        logical_bytes = (contacts_per_step * 352 + revolute_constraints * 400) * iteration_scale
        estimated_flops = (contacts_per_step * 450 + revolute_constraints * 600) * iteration_scale
        roofline_model = "prepared contact/revolute useful-work lower bound; no GPU counters"
    logical_min_gbps = logical_bytes / elapsed / 1.0e9 if logical_bytes is not None else None
    estimated_tflops = estimated_flops / elapsed / 1.0e12 if estimated_flops is not None else None
    return {
        "checkpoint": (
            (
                "C3-serial-world"
                if args.solve_layout == "serial_world"
                else ("C3-shared-body-cache" if args.shared_body_cache else "C2-packed-vec4")
            )
            if packed_contact_path
            else ("C4-packed-mixed" if packed_mixed_path else "R0-revolute-fallback")
        )
        if args.solver == "mini"
        else "phoenx-baseline",
        "solver": args.solver,
        "contact_matching": contact_matching,
        "roofline_basis": roofline_model,
        "device": device.name,
        "scene": args.scene,
        "worlds": args.worlds,
        "bodies_per_world": args.bodies_per_world,
        "contacts_per_step": contacts_per_step,
        "substeps": args.substeps,
        "iterations": args.iterations,
        "block_dim": args.block_dim,
        "max_colors": args.max_colors,
        "elapsed_s": elapsed,
        "frame_us": elapsed * 1.0e6 / args.replays,
        "world_steps_per_s": world_steps / elapsed,
        "body_steps_per_s": body_steps / elapsed,
        "constraint_iterations_per_s": constraint_iteration_rate,
        "grouped_constraints_per_s": grouped_constraint_rate,
        "scalar_rows_per_s": scalar_row_rate,
        "revolute_constraints_per_step": revolute_constraints,
        "logical_min_gbps": logical_min_gbps,
        "sequential_bandwidth_percent": 100.0 * logical_min_gbps / 1489.14 if logical_min_gbps else None,
        "random_scalar_bandwidth_percent": 100.0 * logical_min_gbps / 609.60 if logical_min_gbps else None,
        "random_vec4_bandwidth_percent": 100.0 * logical_min_gbps / 1036.82 if logical_min_gbps else None,
        "estimated_tflops": estimated_tflops,
        "fp32_peak_percent": 100.0 * estimated_tflops / 87.810 if estimated_tflops else None,
        "overflow_constraints": stats.overflow_constraints if stats else None,
        "gather_overflow": stats.gather_overflow if stats else None,
        "color_overflow": stats.color_overflow if stats else None,
        "max_constraints_in_world": stats.max_constraints_in_world if stats else None,
        "total_constraints": stats.total_constraints if stats else None,
        "max_colors_in_world": stats.max_colors_in_world if stats else None,
        "max_constraints_in_color": stats.max_constraints_in_color if stats else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--solver", choices=("mini", "phoenx"), default="mini")
    parser.add_argument("--scene", choices=("stack", "robot"), default="stack")
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--bodies-per-world", type=int, default=8)
    parser.add_argument("--max-contacts-per-world", type=int, default=64)
    parser.add_argument("--max-constraints-per-world", type=int, default=128)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--block-dim", type=int, default=32)
    parser.add_argument("--shared-body-cache", action="store_true")
    parser.add_argument("--solve-layout", choices=("colored", "serial_world"), default="colored")
    parser.add_argument("--phoenx-threads-per-world", choices=("auto", "8", "16", "32"), default="auto")
    parser.add_argument("--max-colors", type=int, default=64)
    parser.add_argument("--max-constraints-per-color", type=int, default=32)
    parser.add_argument("--broad-phase", choices=("nxn", "sap", "explicit"), default="nxn")
    parser.add_argument("--contact-matching", choices=("auto", "disabled", "latest", "sticky"), default="auto")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays", type=int, default=200)
    args = parser.parse_args()
    if args.phoenx_threads_per_world != "auto":
        args.phoenx_threads_per_world = int(args.phoenx_threads_per_world)
    print(json.dumps(_run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
