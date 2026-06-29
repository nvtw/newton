# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure colored-PGS convergence on an ill-conditioned motor chain.

This research benchmark deliberately reuses the exact physical scene from
``example_motorized_hinge_chain`` while exposing solver scheduling knobs. It
runs only on CUDA and always measures replay of an end-to-end captured graph.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.examples import example_motorized_hinge_chain as scene
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _build_world(args: argparse.Namespace, device: wp.context.Device) -> PhoenXWorld:
    bodies = body_container_zeros(scene.NUM_BODIES, device=device)
    scene._populate_chain_bodies(bodies, device)
    constraints = PhoenXWorld.make_constraint_container(num_joints=scene.NUM_HINGES, device=device)
    world = PhoenXWorld(
        bodies=bodies,
        constraints=constraints,
        substeps=args.substeps,
        solver_iterations=args.iterations,
        velocity_iterations=args.velocity_iterations,
        sor_boost=args.sor,
        gravity=(0.0, 0.0, -9.81),
        rigid_contact_max=0,
        num_joints=scene.NUM_HINGES,
        step_layout=args.step_layout,
        symmetric_color_sweep=args.symmetric_sweep,
        mass_splitting=args.mass_splitting,
        max_colored_partitions=args.max_colored_partitions,
        prepare_refresh_stride=args.prepare_refresh_stride,
        cache_articulation_topology=False,
        device=device,
    )
    world.initialize_actuated_double_ball_socket_joints(**scene._build_joint_arrays(device))
    return world


def _metrics(world: PhoenXWorld) -> dict[str, float | int | str | bool]:
    position = world.bodies.position.numpy()[1:]
    velocity = world.bodies.velocity.numpy()[1:]
    angular_velocity = world.bodies.angular_velocity.numpy()[1:]
    pitch, _ = scene._link_layout()
    segment = np.diff(np.vstack((np.zeros((1, 3), dtype=np.float32), position)), axis=0)
    expected_length = np.full(position.shape[0], pitch, dtype=np.float32)
    expected_length[0] = 0.5 * pitch
    segment_error = np.linalg.norm(segment, axis=1) - expected_length
    return {
        "tip_sag_m": float(-position[-1, 2]),
        "max_abs_z_m": float(np.max(np.abs(position[:, 2]))),
        "rms_segment_error_m": float(np.sqrt(np.mean(segment_error * segment_error))),
        "max_segment_error_m": float(np.max(np.abs(segment_error))),
        "rms_speed_m_s": float(np.sqrt(np.mean(np.sum(velocity * velocity, axis=1)))),
        "rms_angular_speed_rad_s": float(np.sqrt(np.mean(np.sum(angular_velocity * angular_velocity, axis=1)))),
        "num_colors": int(world.step_report().num_colors),
    }


def _initialize_static_partition(world: PhoenXWorld, ordering: str) -> None:
    """Build the fixed joint graph once and optionally install chain order."""
    world._rebuild_elements()
    world._partitioner.reset(world._elements, world._num_active_constraints)
    if world.step_layout == "single_world":
        if ordering != "greedy":
            raise ValueError("chain ordering currently requires step_layout=multi_world")
        world._partitioner.build_csr_greedy_with_jp_fallback(
            compute_family_starts=world._singleworld_needs_family_starts()
        )
    else:
        world._build_per_world_coloring()
    if ordering == "chain":
        count = world.num_joints
        ids = world._world_element_ids_by_color.numpy()
        ids[:count] = np.arange(count, dtype=np.int32)
        world._world_element_ids_by_color.assign(ids)
        starts = world._world_color_starts.numpy()
        starts[0, : count + 1] = np.arange(count + 1, dtype=np.int32)
        starts[0, count + 1 :] = count
        world._world_color_starts.assign(starts)
        world._world_num_colors.assign(np.array([count], dtype=np.int32))
    world._dispatcher.begin_step()


def run(args: argparse.Namespace) -> dict[str, float | int | str | bool]:
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("bench_pgs_motor_chain requires CUDA graph capture")
    world = _build_world(args, device)
    _initialize_static_partition(world, args.ordering)

    with wp.ScopedCapture(device=device) as capture:
        world.step(args.dt, reuse_partition=True)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    start = time.perf_counter()
    for _ in range(args.frames):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - start

    result = _metrics(world)
    result.update(
        {
            "frames": args.frames,
            "dt_s": args.dt,
            "substeps": args.substeps,
            "solver_iterations": args.iterations,
            "velocity_iterations": args.velocity_iterations,
            "sor": args.sor,
            "symmetric_sweep": args.symmetric_sweep,
            "mass_splitting": args.mass_splitting,
            "max_colored_partitions": args.max_colored_partitions,
            "step_layout": args.step_layout,
            "ordering": args.ordering,
            "elapsed_s": elapsed,
            "frame_time_ms": 1000.0 * elapsed / args.frames,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--substeps", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--sor", type=float, default=1.0)
    parser.add_argument("--step-layout", choices=("single_world", "multi_world"), default="multi_world")
    parser.add_argument("--ordering", choices=("greedy", "chain"), default="greedy")
    parser.add_argument("--symmetric-sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mass-splitting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-colored-partitions", type=int, default=12)
    parser.add_argument("--prepare-refresh-stride", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
