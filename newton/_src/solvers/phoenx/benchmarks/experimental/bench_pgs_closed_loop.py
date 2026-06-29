# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph benchmark for PGS on a motorized closed revolute loop."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

import newton


def _build_model(link_count: int, radius: float, contacts: bool) -> newton.Model:
    builder = newton.ModelBuilder()
    if contacts:
        builder.add_ground_plane()
    angles = 2.0 * np.pi * np.arange(link_count) / link_count
    vertices = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles), np.full(link_count, 0.035 if contacts else 0.0))
    ).astype(np.float32)
    bodies: list[int] = []
    length = float(2.0 * radius * np.sin(np.pi / link_count))
    for link in range(link_count):
        start = vertices[link]
        end = vertices[(link + 1) % link_count]
        center = 0.5 * (start + end)
        direction = end - start
        yaw = float(np.arctan2(direction[1], direction[0]))
        body = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(*center),
                q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
            ),
        )
        builder.add_shape_box(
            body,
            hx=0.5 * length,
            hy=0.035,
            hz=0.035,
            cfg=newton.ModelBuilder.ShapeConfig(density=1200.0, mu=0.6),
        )
        bodies.append(body)

    anchor = builder.add_joint_revolute(
        parent=-1,
        child=bodies[0],
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(
            p=wp.vec3(*np.asarray(builder.body_q[bodies[0]].p)),
            q=wp.quat_identity(),
        ),
        child_xform=wp.transform_identity(),
        target_pos=0.35,
        target_ke=200.0,
        target_kd=20.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    tree_joints = [anchor]
    for child in range(1, link_count):
        joint = builder.add_joint_revolute(
            parent=bodies[child - 1],
            child=bodies[child],
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(0.5 * length, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-0.5 * length, 0.0, 0.0), q=wp.quat_identity()),
        )
        tree_joints.append(joint)
    builder.add_joint_revolute(
        parent=bodies[-1],
        child=bodies[0],
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(0.5 * length, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-0.5 * length, 0.0, 0.0), q=wp.quat_identity()),
    )
    builder.add_articulation(tree_joints)
    builder.gravity = 0.0
    return builder.finalize()


def run(args: argparse.Namespace) -> dict[str, float | int | bool | str]:
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("closed-loop PGS benchmark requires CUDA graph capture")
    model = _build_model(args.links, args.radius, args.contacts)
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=args.substeps,
        solver_iterations=args.iterations,
        velocity_iterations=args.velocity_iterations,
        articulation_coarse_mode="auto",
        articulation_coarse_stride=args.coarse_stride,
        articulation_coarse_color_sweeps=args.coarse_color_sweeps,
        articulation_coarse_regularization=args.coarse_regularization,
    )
    if not args.coarse:
        solver.world.articulation_dvi_host = False
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = None
    if args.contacts:
        collision_pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        contacts = model.contacts(collision_pipeline=collision_pipeline)

    def step_pair() -> None:
        nonlocal state_0, state_1
        for _ in range(2):
            state_0.clear_forces()
            if contacts is not None:
                model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, args.dt)
            state_0, state_1 = state_1, state_0

    step_pair()
    with wp.ScopedCapture(device=device) as capture:
        step_pair()
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(args.frames // 2):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - start

    velocity = state_0.body_qd.numpy()
    setup = solver.world.articulation_coarse_setup
    system = solver.world.articulation_device_system
    system.populate_from_adbs_constraints(
        solver.world.constraints,
        solver.world.bodies,
        dt=solver.world.substep_dt,
        device=device,
    )
    violation = system.violation.numpy()[: system.total_rows].reshape(-1, 5)
    position_violation = np.linalg.norm(violation[:, :3], axis=1)
    angular_violation = np.linalg.norm(violation[:, 3:], axis=1)
    result = {
        "coarse": args.coarse,
        "contacts": args.contacts,
        "resolved_mode": "none" if setup is None else setup.mode,
        "links": args.links,
        "frames": args.frames,
        "substeps": args.substeps,
        "iterations": args.iterations,
        "frame_time_ms": 1000.0 * elapsed / args.frames,
        "rms_position_violation_m": float(np.sqrt(np.mean(position_violation * position_violation))),
        "max_position_violation_m": float(np.max(position_violation)),
        "rms_angular_violation_rad": float(np.sqrt(np.mean(angular_violation * angular_violation))),
        "rms_speed_m_s": float(np.sqrt(np.mean(np.sum(velocity[:, :3] * velocity[:, :3], axis=1)))),
    }
    if contacts is not None:
        result["contact_count"] = int(contacts.rigid_contact_count.numpy()[0])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--links", type=int, default=32)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--substeps", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--coarse", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--contacts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coarse-stride", type=int, default=2)
    parser.add_argument("--coarse-color-sweeps", type=int, default=16)
    parser.add_argument("--coarse-regularization", type=float, default=0.001)
    args = parser.parse_args()
    if args.frames < 2 or args.frames % 2:
        parser.error("--frames must be an even integer of at least two")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
