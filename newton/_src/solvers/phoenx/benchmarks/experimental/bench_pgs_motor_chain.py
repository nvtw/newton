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

from newton._src.solvers.phoenx.articulations.device import ArticulationDeviceSystem
from newton._src.solvers.phoenx.articulations.symbolic import compute_block_sparse_symbolic
from newton._src.solvers.phoenx.benchmarks.experimental.balanced_sparse_solve import (
    solve_balanced_sparse_factors,
    solve_balanced_sparse_matrix,
)
from newton._src.solvers.phoenx.benchmarks.experimental.block_path_solve import solve_block_path_matrix
from newton._src.solvers.phoenx.benchmarks.experimental.coarse_path_solve import CoarsePathSolver
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
        cache_articulation_topology=True,
        articulation_dvi_host=args.dvi_every_substep,
        articulation_dvi_replaces_joint_pgs=False,
        articulation_dvi_host_solver="device_block_sparse",
        articulation_dvi_stride=args.dvi_stride,
        articulation_dvi_relaxation=args.dvi_relaxation,
        device=device,
    )
    world.initialize_actuated_double_ball_socket_joints(**scene._build_joint_arrays(device))
    if args.global_corrections > 0 or args.dvi_every_substep:
        topology = world.articulation_topology
        if topology is None:
            raise RuntimeError("failed to cache articulation topology")
        symbolic = compute_block_sparse_symbolic(
            topology.active_body1,
            topology.active_body2,
            topology.active_row_counts,
            use_meca=not (args.dvi_path_megakernel or args.dvi_coarse_sgs),
            use_parallel_path_ordering=not (args.dvi_path_megakernel or args.dvi_coarse_sgs),
        )
        world.articulation_device_system = ArticulationDeviceSystem.from_topology(topology, device, symbolic)
        system = world.articulation_device_system
        if args.dvi_path_megakernel:
            system.solve_block_sparse_matrix = lambda *, device=None: solve_block_path_matrix(system, device=device)
        elif args.dvi_coarse_sgs:
            coarse_solver = CoarsePathSolver(
                system.block_count,
                int(topology.active_row_counts[0]),
                args.dvi_coarse_color_sweeps,
                device,
            )
            system.solve_block_sparse_matrix = lambda *, device=None: coarse_solver.solve(system, device=device)
        elif args.dvi_fused_balanced:
            system.solve_block_sparse_matrix = lambda *, device=None: solve_balanced_sparse_matrix(
                system, device=device
            )
        if world.articulation_system is None:
            raise RuntimeError("failed to cache articulation system")
        world.articulation_system.diagonal_regularization = args.dvi_regularization
        if args.dvi_factor_refresh > 1:
            if args.dvi_path_megakernel or args.dvi_coarse_sgs:
                raise ValueError("factor reuse does not support natural-path experimental solvers")

            def solve_with_reused_factors(
                dt=None,
                *,
                alpha=0.0,
                recovery_speed=-1.0,
                solver=None,
            ):
                del solver
                solve_dt = world.substep_dt if dt is None else float(dt)
                system.populate_from_adbs_constraints(world.constraints, world.bodies, dt=solve_dt, device=device)
                system.compute_residual(
                    world.bodies,
                    dt=solve_dt,
                    alpha=float(alpha),
                    recovery_speed=float(recovery_speed),
                    device=device,
                )
                if world._current_substep_index % args.dvi_factor_refresh == 0:
                    system.assemble_block_sparse_matrix(
                        world.bodies.inverse_mass,
                        world.bodies.inverse_inertia_world,
                        diagonal_regularization=args.dvi_regularization,
                        device=device,
                    )
                    system.factor_block_sparse_matrix(device=device)
                if args.dvi_fused_balanced:
                    solve_balanced_sparse_factors(system, device=device)
                else:
                    system.gather_block_rhs(device=device)
                    system.solve_block_sparse_factors(device=device)
                    system.scatter_block_solution(device=device)
                system.apply_solution(
                    world.bodies,
                    world.bodies.inverse_mass,
                    world.bodies.inverse_inertia_world,
                    solution_scale=args.dvi_relaxation,
                    device=device,
                )
                return True

            world.solve_articulations_dvi_host = solve_with_reused_factors
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
    system = world.articulation_device_system
    if system is None:
        raise RuntimeError("motor-chain metrics require cached articulation rows")
    system.populate_from_adbs_constraints(
        world.constraints,
        world.bodies,
        dt=world.substep_dt,
        device=world.device,
    )
    violation = system.violation.numpy()[: system.total_rows].reshape(scene.NUM_HINGES, 5)
    position_violation = np.linalg.norm(violation[:, :3], axis=1)
    angular_violation = np.linalg.norm(violation[:, 3:], axis=1)
    return {
        "tip_sag_m": float(-position[-1, 2]),
        "max_abs_z_m": float(np.max(np.abs(position[:, 2]))),
        "rms_segment_error_m": float(np.sqrt(np.mean(segment_error * segment_error))),
        "max_segment_error_m": float(np.max(np.abs(segment_error))),
        "rms_joint_position_violation_m": float(np.sqrt(np.mean(position_violation * position_violation))),
        "max_joint_position_violation_m": float(np.max(position_violation)),
        "rms_joint_angular_violation_rad": float(np.sqrt(np.mean(angular_violation * angular_violation))),
        "max_joint_angular_violation_rad": float(np.max(angular_violation)),
        "rms_speed_m_s": float(np.sqrt(np.mean(np.sum(velocity * velocity, axis=1)))),
        "rms_angular_speed_rad_s": float(np.sqrt(np.mean(np.sum(angular_velocity * angular_velocity, axis=1)))),
        "num_colors": int(world.step_report().num_colors),
    }


def _initialize_static_partition(world: PhoenXWorld, ordering: str, stripe_width: int) -> None:
    """Build the fixed joint graph once and optionally install chain order."""
    world._rebuild_elements()
    world._partitioner.reset(world._elements, world._num_active_constraints)
    if world.step_layout == "single_world":
        if ordering != "greedy":
            raise ValueError("custom ordering currently requires step_layout=multi_world")
        world._partitioner.build_csr_greedy_with_jp_fallback(
            compute_family_starts=world._singleworld_needs_family_starts()
        )
    else:
        world._build_per_world_coloring()
    if ordering != "greedy":
        count = world.num_joints
        width = count if ordering == "chain" else stripe_width
        if width < 2 or width > count:
            raise ValueError(f"stripe width must be in [2, {count}], got {width}")
        color = np.arange(count, dtype=np.int32) % width
        permutation = np.argsort(color, kind="stable").astype(np.int32)
        ids = world._world_element_ids_by_color.numpy()
        ids[:count] = permutation
        world._world_element_ids_by_color.assign(ids)
        starts = world._world_color_starts.numpy()
        color_count = np.bincount(color, minlength=width)
        starts[0, : width + 1] = np.concatenate(([0], np.cumsum(color_count))).astype(np.int32)
        starts[0, width + 1 :] = count
        world._world_color_starts.assign(starts)
        world._world_num_colors.assign(np.array([width], dtype=np.int32))
    world._dispatcher.begin_step()


def run(args: argparse.Namespace) -> dict[str, float | int | str | bool]:
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("bench_pgs_motor_chain requires CUDA graph capture")
    world = _build_world(args, device)
    _initialize_static_partition(world, args.ordering, args.stripe_width)

    correction_dt = args.dt / args.substeps
    with wp.ScopedCapture(device=device) as capture:
        for _ in range(args.global_corrections):
            system = world.articulation_device_system
            if system is None:
                raise RuntimeError("global correction system is unavailable")
            system.populate_from_adbs_constraints(world.constraints, world.bodies, dt=correction_dt, device=device)
            system.compute_residual(
                world.bodies,
                dt=correction_dt,
                recovery_speed=0.0,
                device=device,
            )
            system.assemble_block_sparse_matrix(
                world.bodies.inverse_mass,
                world.bodies.inverse_inertia_world,
                diagonal_regularization=args.global_regularization,
                device=device,
            )
            system.solve_block_sparse_matrix(device=device)
            system.apply_solution(
                world.bodies,
                world.bodies.inverse_mass,
                world.bodies.inverse_inertia_world,
                device=device,
            )
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
            "stripe_width": args.stripe_width,
            "global_corrections": args.global_corrections,
            "global_regularization": args.global_regularization,
            "dvi_every_substep": args.dvi_every_substep,
            "dvi_stride": args.dvi_stride,
            "dvi_regularization": args.dvi_regularization,
            "dvi_relaxation": args.dvi_relaxation,
            "dvi_path_megakernel": args.dvi_path_megakernel,
            "dvi_fused_balanced": args.dvi_fused_balanced,
            "dvi_factor_refresh": args.dvi_factor_refresh,
            "dvi_coarse_sgs": args.dvi_coarse_sgs,
            "dvi_coarse_color_sweeps": args.dvi_coarse_color_sweeps,
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
    parser.add_argument("--ordering", choices=("greedy", "striped", "chain"), default="greedy")
    parser.add_argument("--stripe-width", type=int, default=8)
    parser.add_argument("--symmetric-sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mass-splitting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-colored-partitions", type=int, default=12)
    parser.add_argument("--prepare-refresh-stride", type=int, default=1)
    parser.add_argument("--global-corrections", type=int, default=0)
    parser.add_argument("--global-regularization", type=float, default=0.1)
    parser.add_argument("--dvi-every-substep", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dvi-stride", type=int, default=1)
    parser.add_argument("--dvi-regularization", type=float, default=1.0e-4)
    parser.add_argument("--dvi-relaxation", type=float, default=1.0)
    parser.add_argument("--dvi-path-megakernel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dvi-fused-balanced", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dvi-factor-refresh", type=int, default=1)
    parser.add_argument("--dvi-coarse-sgs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dvi-coarse-color-sweeps", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
