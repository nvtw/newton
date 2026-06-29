# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph benchmark for PGS on a loaded branched body tree."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.experimental.analyze_pgs_branched_tree import _tree_edges
from newton._src.solvers.phoenx.benchmarks.experimental.bench_pgs_motor_chain import (
    _initialize_static_partition,
)
from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, MOTION_STATIC, body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_joint import JOINT_MODE_BALL_SOCKET
from newton._src.solvers.phoenx.examples import example_motorized_hinge_chain as motor_scene
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.phoenx.tests.test_articulation_dvi import _make_adbs_world


def _make_realistic_world(args: argparse.Namespace, device, body1, body2, depth):
    body_count = int(body2.max()) + 1
    bodies = body_container_zeros(body_count, device=device)
    template_bodies = body_container_zeros(motor_scene.NUM_BODIES, device=device)
    motor_scene._populate_chain_bodies(template_bodies, device)
    pitch, orientation = motor_scene._link_layout()
    positions = np.zeros((body_count, 3), dtype=np.float32)
    orientations = np.zeros((body_count, 4), dtype=np.float32)
    orientations[0] = (0.0, 0.0, 0.0, 1.0)
    for joint, child in enumerate(body2):
        positions[child] = (0.0, -(depth[joint] - 0.5) * pitch, 0.0)
        orientations[child] = orientation
    bodies.position.assign(positions)
    bodies.orientation.assign(orientations)
    for name in ("inverse_mass", "inverse_inertia", "inverse_inertia_world"):
        source = getattr(template_bodies, name).numpy()
        values = np.repeat(source[1:2], body_count, axis=0)
        values[0] = source[0]
        getattr(bodies, name).assign(values)
    motion = np.full(body_count, int(MOTION_DYNAMIC), dtype=np.int32)
    motion[0] = int(MOTION_STATIC)
    bodies.motion_type.assign(motion)

    joint_count = int(body1.size)
    constraints = PhoenXWorld.make_constraint_container(num_joints=joint_count, device=device)
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        num_joints=joint_count,
        substeps=args.substeps,
        solver_iterations=args.iterations,
        velocity_iterations=args.velocity_iterations,
        gravity=(0.0, 0.0, -9.81),
        rigid_contact_max=0,
        cache_articulation_topology=True,
        articulation_dvi_host=args.coarse,
        articulation_dvi_replaces_joint_pgs=False,
        articulation_dvi_host_solver="device_block_sparse",
        articulation_dvi_stride=args.coarse_stride,
        articulation_coarse_mode="tree" if args.coarse else None,
        articulation_coarse_stride=args.coarse_stride,
        articulation_coarse_color_sweeps=args.coarse_color_sweeps,
        articulation_coarse_regularization=args.coarse_regularization,
        device=device,
    )
    anchor_offset = motor_scene._CAPSULE_RADIUS
    anchor1 = np.zeros((joint_count, 3), dtype=np.float32)
    anchor2 = np.zeros((joint_count, 3), dtype=np.float32)
    for joint in range(joint_count):
        y = -(depth[joint] - 1) * pitch
        anchor1[joint] = (0.0, y, -anchor_offset)
        anchor2[joint] = (0.0, y, anchor_offset)
    template = motor_scene._build_joint_arrays(device)
    joint_arrays = {
        "body1": wp.array(body1, dtype=wp.int32, device=device),
        "body2": wp.array(body2, dtype=wp.int32, device=device),
        "anchor1": wp.array(anchor1, dtype=wp.vec3f, device=device),
        "anchor2": wp.array(anchor2, dtype=wp.vec3f, device=device),
    }
    for name, array in template.items():
        if name in joint_arrays:
            continue
        source = array.numpy()
        values = np.repeat(source[:1], joint_count, axis=0)
        joint_arrays[name] = wp.array(values, dtype=array.dtype, device=device)
    world.initialize_actuated_double_ball_socket_joints(**joint_arrays)
    return world


def _build_world(args: argparse.Namespace, device):
    body1, body2, depth = _tree_edges(args.trunk, args.arms, args.arm_length)
    if args.realistic_links:
        world = _make_realistic_world(args, device, body1, body2, depth)
    else:
        positions = np.zeros((int(body2.max()) + 1, 3), dtype=np.float32)
        world = _make_adbs_world(
            device,
            body1,
            body2,
            np.full(body1.size, int(JOINT_MODE_BALL_SOCKET), dtype=np.int32),
            positions_np=positions,
            world_kwargs={
                "substeps": args.substeps,
                "solver_iterations": args.iterations,
                "velocity_iterations": args.velocity_iterations,
                "gravity": (0.0, 0.0, -9.81),
                "cache_articulation_topology": True,
                "articulation_dvi_host": args.coarse,
                "articulation_dvi_replaces_joint_pgs": False,
                "articulation_dvi_host_solver": "device_block_sparse",
                "articulation_dvi_stride": args.coarse_stride,
                "articulation_coarse_mode": "tree" if args.coarse else None,
                "articulation_coarse_stride": args.coarse_stride,
                "articulation_coarse_color_sweeps": args.coarse_color_sweeps,
                "articulation_coarse_regularization": args.coarse_regularization,
            },
        )
    return world


def _metrics(world) -> dict[str, float]:
    position = world.bodies.position.numpy()[1:]
    velocity = world.bodies.velocity.numpy()[1:]
    system = world.articulation_device_system
    system.populate_from_adbs_constraints(
        world.constraints,
        world.bodies,
        dt=world.substep_dt,
        device=world.device,
    )
    rows = system.total_rows // world.num_joints
    violation = system.violation.numpy()[: system.total_rows].reshape(-1, rows)
    violation_norm = np.linalg.norm(violation[:, :3], axis=1)
    angular_violation = np.linalg.norm(violation[:, 3:], axis=1) if rows > 3 else np.zeros_like(violation_norm)
    return {
        "max_sag_m": float(np.max(-position[:, 2])),
        "rms_displacement_m": float(np.sqrt(np.mean(np.sum(position * position, axis=1)))),
        "rms_speed_m_s": float(np.sqrt(np.mean(np.sum(velocity * velocity, axis=1)))),
        "rms_joint_violation_m": float(np.sqrt(np.mean(violation_norm * violation_norm))),
        "max_joint_violation_m": float(np.max(violation_norm)),
        "rms_joint_angular_violation_rad": float(np.sqrt(np.mean(angular_violation * angular_violation))),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("branched-tree benchmark requires CUDA graph capture")
    world = _build_world(args, device)
    _initialize_static_partition(world, "greedy", 2)
    with wp.ScopedCapture(device=device) as capture:
        world.step(args.dt, reuse_partition=True)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(args.frames):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - start
    result: dict[str, object] = _metrics(world)
    result.update(
        {
            "frames": args.frames,
            "substeps": args.substeps,
            "iterations": args.iterations,
            "coarse": args.coarse,
            "coarse_stride": args.coarse_stride,
            "coarse_color_sweeps": args.coarse_color_sweeps,
            "coarse_regularization": args.coarse_regularization,
            "frame_time_ms": 1000.0 * elapsed / args.frames,
            "joint_count": world.num_joints,
            "realistic_links": args.realistic_links,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--trunk", type=int, default=32)
    parser.add_argument("--arms", type=int, default=3)
    parser.add_argument("--arm-length", type=int, default=24)
    parser.add_argument("--substeps", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--coarse", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coarse-stride", type=int, default=2)
    parser.add_argument("--coarse-color-sweeps", type=int, default=16)
    parser.add_argument("--coarse-regularization", type=float, default=0.001)
    parser.add_argument("--realistic-links", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
