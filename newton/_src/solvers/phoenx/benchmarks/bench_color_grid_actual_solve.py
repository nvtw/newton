# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental PhoenX global-color solve benchmark.

The production multi-world fast-tail solver assigns a small lane group to each
world and loops colors inside one kernel. That has low launch overhead, but few
worlds can leave many SMs idle. This benchmark keeps the same color ordering and
PGS safety, but launches each color as one global grid over all worlds' rows.
It is aimed at single/few-world scenes with enough rows per color to fill the
GPU.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _extract_solver
from newton._src.solvers.phoenx.benchmarks.scenarios import g1_flat, h1_flat, tower
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    actuated_double_ball_socket_iterate_multi,
    actuated_double_ball_socket_prepare_for_iteration,
    revolute_iterate_multi,
    revolute_prepare_for_iteration,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_iterate_multi_no_soft_pd,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import (
    contact_prepare_for_iteration_lean_no_soft_pd,
)
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.phoenx.solver_phoenx_kernels import _FUSED_INNER_SWEEPS


@dataclass
class ColorGridHost:
    eids: np.ndarray
    color_starts: np.ndarray
    color_max_counts: np.ndarray
    num_colors: int


@dataclass
class ColorGridDevice:
    host: ColorGridHost
    eids: wp.array
    color_starts: wp.array
    processed: wp.array


@wp.kernel(enable_backward=False)
def _color_grid_reset_kernel(processed: wp.array[wp.int32]):
    processed[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _color_grid_prepare_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    eids: wp.array[wp.int32],
    row_start: wp.int32,
    row_count: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    processed: wp.array[wp.int32],
    copy_state: CopyStateContainer,
):
    tid = wp.tid()
    if tid < row_count:
        cid = eids[row_start + tid]
        if cid < num_joints:
            if revolute_only != wp.int32(0):
                revolute_prepare_for_iteration(
                    constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                )
            else:
                actuated_double_ball_socket_prepare_for_iteration(
                    constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                )
        else:
            contact_prepare_for_iteration_lean_no_soft_pd(
                contact_cols,
                cid - num_joints,
                bodies,
                particles,
                num_bodies,
                idt,
                cc,
                contacts,
                copy_state,
                wp.int32(0),
            )


@wp.kernel(enable_backward=False)
def _color_grid_iterate_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    eids: wp.array[wp.int32],
    row_start: wp.int32,
    row_count: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    inner_sweeps: wp.int32,
    processed: wp.array[wp.int32],
    copy_state: CopyStateContainer,
):
    tid = wp.tid()
    if tid < row_count:
        cid = eids[row_start + tid]
        if cid < num_joints:
            if revolute_only != wp.int32(0):
                revolute_iterate_multi(
                    constraints,
                    cid,
                    bodies,
                    particles,
                    copy_state,
                    num_bodies,
                    wp.int32(0),
                    idt,
                    sor_boost,
                    True,
                    inner_sweeps,
                )
            else:
                actuated_double_ball_socket_iterate_multi(
                    constraints,
                    cid,
                    bodies,
                    particles,
                    copy_state,
                    num_bodies,
                    wp.int32(0),
                    idt,
                    sor_boost,
                    True,
                    inner_sweeps,
                )
        else:
            contact_iterate_multi_no_soft_pd(
                contact_cols,
                cid - num_joints,
                bodies,
                particles,
                num_bodies,
                idt,
                cc,
                contacts,
                True,
                inner_sweeps,
                copy_state,
                wp.int32(0),
                sor_boost,
            )


@wp.kernel(enable_backward=False)
def _color_grid_prepare_direct_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    color: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    copy_state: CopyStateContainer,
):
    world_id, local_row = wp.tid()
    if color < world_num_colors[world_id]:
        world_base = world_csr_offsets[world_id]
        start = world_base + world_color_starts[world_id, color]
        end = world_base + world_color_starts[world_id, color + wp.int32(1)]
        if local_row < end - start:
            cid = world_element_ids_by_color[start + local_row]
            if cid < num_joints:
                if revolute_only != wp.int32(0):
                    revolute_prepare_for_iteration(
                        constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                    )
                else:
                    actuated_double_ball_socket_prepare_for_iteration(
                        constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                    )
            else:
                contact_prepare_for_iteration_lean_no_soft_pd(
                    contact_cols,
                    cid - num_joints,
                    bodies,
                    particles,
                    num_bodies,
                    idt,
                    cc,
                    contacts,
                    copy_state,
                    wp.int32(0),
                )


@wp.kernel(enable_backward=False)
def _color_grid_iterate_direct_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    color: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    inner_sweeps: wp.int32,
    copy_state: CopyStateContainer,
):
    world_id, local_row = wp.tid()
    if color < world_num_colors[world_id]:
        world_base = world_csr_offsets[world_id]
        start = world_base + world_color_starts[world_id, color]
        end = world_base + world_color_starts[world_id, color + wp.int32(1)]
        if local_row < end - start:
            cid = world_element_ids_by_color[start + local_row]
            if cid < num_joints:
                if revolute_only != wp.int32(0):
                    revolute_iterate_multi(
                        constraints,
                        cid,
                        bodies,
                        particles,
                        copy_state,
                        num_bodies,
                        wp.int32(0),
                        idt,
                        sor_boost,
                        True,
                        inner_sweeps,
                    )
                else:
                    actuated_double_ball_socket_iterate_multi(
                        constraints,
                        cid,
                        bodies,
                        particles,
                        copy_state,
                        num_bodies,
                        wp.int32(0),
                        idt,
                        sor_boost,
                        True,
                        inner_sweeps,
                    )
            else:
                contact_iterate_multi_no_soft_pd(
                    contact_cols,
                    cid - num_joints,
                    bodies,
                    particles,
                    num_bodies,
                    idt,
                    cc,
                    contacts,
                    True,
                    inner_sweeps,
                    copy_state,
                    wp.int32(0),
                    sor_boost,
                )


def _build_scene(scene: str, num_worlds: int, *, substeps: int, solver_iterations: int):
    if scene == "h1":
        return h1_flat.build(num_worlds, "phoenx", substeps, solver_iterations)
    if scene == "g1":
        return g1_flat.build(num_worlds, "phoenx", substeps, solver_iterations)
    if scene == "tower":
        return tower.build(num_worlds, "phoenx", substeps, solver_iterations, step_layout="multi_world")
    raise ValueError(f"unknown scene: {scene}")


def _extract_color_grid(world: PhoenXWorld) -> ColorGridHost:
    active = int(world._num_active_constraints.numpy()[0])
    eids_by_color = world._world_element_ids_by_color.numpy()
    starts = world._world_color_starts.numpy()
    csr = world._world_csr_offsets.numpy()
    num_colors_per_world = world._world_num_colors.numpy().astype(np.int32, copy=False)
    num_colors = int(num_colors_per_world.max(initial=0))
    eids: list[int] = []
    color_starts = np.zeros(num_colors + 1, dtype=np.int32)
    color_max_counts = np.zeros(num_colors, dtype=np.int32)
    for color in range(num_colors):
        color_starts[color] = len(eids)
        for world_id in range(world.num_worlds):
            if color >= int(num_colors_per_world[world_id]):
                continue
            base = int(csr[world_id])
            start = base + int(starts[world_id, color])
            end = base + int(starts[world_id, color + 1])
            color_max_counts[color] = max(int(color_max_counts[color]), end - start)
            for cursor in range(start, end):
                eid = int(eids_by_color[cursor])
                if 0 <= eid < active:
                    eids.append(eid)
    color_starts[num_colors] = len(eids)
    if not eids:
        eids.append(0)
    return ColorGridHost(
        eids=np.asarray(eids, dtype=np.int32),
        color_starts=color_starts,
        color_max_counts=color_max_counts,
        num_colors=num_colors,
    )


def _upload_color_grid(graph: ColorGridHost, device: wp.context.Devicelike) -> ColorGridDevice:
    return ColorGridDevice(
        host=graph,
        eids=wp.array(graph.eids, dtype=wp.int32, device=device),
        color_starts=wp.array(graph.color_starts, dtype=wp.int32, device=device),
        processed=wp.zeros(1, dtype=wp.int32, device=device),
    )


def _color_grid_runner(world: PhoenXWorld, graph: ColorGridDevice, *, block_dim: int):
    device = world.device
    contact_views = world._contact_views if world._contact_views is not None else world._contact_views_placeholder
    idt = wp.float32(1.0 / world.substep_dt)
    inner_sweeps = int(_FUSED_INNER_SWEEPS)
    outer_iters = int(world.solver_iterations) // inner_sweeps
    revolute_only = 1 if bool(world._use_revolute_specialization) else 0
    starts = graph.host.color_starts
    counts = [int(starts[i + 1] - starts[i]) for i in range(graph.host.num_colors)]

    def run() -> None:
        wp.launch(_color_grid_reset_kernel, dim=1, inputs=[graph.processed], device=device)
        for color, count in enumerate(counts):
            if count <= 0:
                continue
            wp.launch(
                _color_grid_prepare_kernel,
                dim=count,
                block_dim=block_dim,
                inputs=[
                    world.constraints,
                    world._contact_cols,
                    world.bodies,
                    world._particles_or_sentinel(),
                    idt,
                    graph.eids,
                    wp.int32(int(starts[color])),
                    wp.int32(count),
                    world._contact_container,
                    contact_views,
                    wp.int32(world.num_joints),
                    wp.int32(world.num_bodies),
                    wp.int32(revolute_only),
                    graph.processed,
                    world._copy_state,
                ],
                device=device,
            )
        for _outer in range(outer_iters):
            for color, count in enumerate(counts):
                if count <= 0:
                    continue
                wp.launch(
                    _color_grid_iterate_kernel,
                    dim=count,
                    block_dim=block_dim,
                    inputs=[
                        world.constraints,
                        world._contact_cols,
                        world.bodies,
                        world._particles_or_sentinel(),
                        idt,
                        wp.float32(world.sor_boost),
                        graph.eids,
                        wp.int32(int(starts[color])),
                        wp.int32(count),
                        world._contact_container,
                        contact_views,
                        wp.int32(world.num_joints),
                        wp.int32(world.num_bodies),
                        wp.int32(revolute_only),
                        wp.int32(inner_sweeps),
                        graph.processed,
                        world._copy_state,
                    ],
                    device=device,
                )

    return run


def _color_grid_direct_runner(world: PhoenXWorld, graph: ColorGridDevice, *, block_dim: int):
    device = world.device
    contact_views = world._contact_views if world._contact_views is not None else world._contact_views_placeholder
    idt = wp.float32(1.0 / world.substep_dt)
    inner_sweeps = int(_FUSED_INNER_SWEEPS)
    outer_iters = int(world.solver_iterations) // inner_sweeps
    revolute_only = 1 if bool(world._use_revolute_specialization) else 0
    max_counts = [int(v) for v in graph.host.color_max_counts]

    def run() -> None:
        for color, max_count in enumerate(max_counts):
            if max_count <= 0:
                continue
            wp.launch(
                _color_grid_prepare_direct_kernel,
                dim=(world.num_worlds, max_count),
                block_dim=block_dim,
                inputs=[
                    world.constraints,
                    world._contact_cols,
                    world.bodies,
                    world._particles_or_sentinel(),
                    idt,
                    world._world_element_ids_by_color,
                    world._world_color_starts,
                    world._world_csr_offsets,
                    world._world_num_colors,
                    wp.int32(color),
                    world._contact_container,
                    contact_views,
                    wp.int32(world.num_joints),
                    wp.int32(world.num_bodies),
                    wp.int32(revolute_only),
                    world._copy_state,
                ],
                device=device,
            )
        for _outer in range(outer_iters):
            for color, max_count in enumerate(max_counts):
                if max_count <= 0:
                    continue
                wp.launch(
                    _color_grid_iterate_direct_kernel,
                    dim=(world.num_worlds, max_count),
                    block_dim=block_dim,
                    inputs=[
                        world.constraints,
                        world._contact_cols,
                        world.bodies,
                        world._particles_or_sentinel(),
                        idt,
                        wp.float32(world.sor_boost),
                        world._world_element_ids_by_color,
                        world._world_color_starts,
                        world._world_csr_offsets,
                        world._world_num_colors,
                        wp.int32(color),
                        world._contact_container,
                        contact_views,
                        wp.int32(world.num_joints),
                        wp.int32(world.num_bodies),
                        wp.int32(revolute_only),
                        wp.int32(inner_sweeps),
                        world._copy_state,
                    ],
                    device=device,
                )

    return run


def _bench(fn, *, n_runs: int, warmup: int, trials: int, device: wp.context.Devicelike) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    wp.synchronize_device()
    with wp.ScopedCapture(device=device) as capture:
        fn()
    graph = capture.graph
    wp.synchronize_device()
    times: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            wp.capture_launch(graph)
        wp.synchronize_device()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return float(arr.min()), float(np.median(arr))


def _validate(graph: ColorGridDevice, expected: int) -> None:
    # The timed kernels intentionally avoid per-row validation atomics.
    # Successful kernel completion is enough for this isolated scheduler bench.
    _ = graph, expected


def run_case(args: argparse.Namespace, scene: str, num_worlds: int) -> None:
    handle = _build_scene(scene, num_worlds, substeps=args.substeps, solver_iterations=args.solver_iterations)
    solver = _extract_solver(handle)
    world = solver.world
    if world.step_layout == "single_world" or world.mass_splitting_enabled or world.num_particles > 0:
        raise RuntimeError("color-grid prototype only supports multi_world rigid scenes")
    if world._has_soft_contact_pd or world._sleeping_enabled:
        raise RuntimeError("color-grid prototype currently excludes soft-contact PD and sleeping")
    for _ in range(args.prime_frames):
        handle.simulate_one_frame()
    wp.synchronize_device()

    host_graph = _extract_color_grid(world)
    grid_graph = _upload_color_grid(host_graph, world.device)
    outer_iters = int(world.solver_iterations) // int(_FUSED_INNER_SWEEPS)
    expected = int(host_graph.color_starts[-1]) * (1 + outer_iters)

    runs: list[tuple[str, object]] = []
    if args.mode in ("flat", "both"):
        runs.append(("flat", _color_grid_runner(world, grid_graph, block_dim=args.block_dim)))
    if args.mode in ("direct", "both"):
        runs.append(("direct", _color_grid_direct_runner(world, grid_graph, block_dim=args.block_dim)))

    for _label, run in runs:
        run()
        wp.synchronize_device()
        _validate(grid_graph, expected)

    base_min, base_med = _bench(
        world._solve_main, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=world.device
    )
    pieces = []
    for label, run in runs:
        min_ms, med_ms = _bench(run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=world.device)
        speed = base_min / min_ms if min_ms > 0.0 else float("nan")
        pieces.append(f"{label}={min_ms:8.3f}ms({speed:5.3f}x)")
        if args.verbose:
            print(f"  {label:6s} min={min_ms:.3f} ms med={med_ms:.3f} ms")
    print(
        f"{scene:5s} worlds={num_worlds:5d} rows={int(host_graph.color_starts[-1]):6d} colors={host_graph.num_colors:4d} "
        f"baseline={base_min:8.3f}ms " + " ".join(pieces) + f" base_us={1000.0 * base_min / args.n_runs:8.3f}"
    )
    if args.verbose:
        print(f"  baseline min={base_min:.3f} ms med={base_med:.3f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes", nargs="+", choices=("h1", "g1", "tower"), default=["h1", "g1", "tower"])
    parser.add_argument("--worlds", default="32", help="Comma-separated world counts.")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--prime-frames", type=int, default=3)
    parser.add_argument("--block-dim", type=int, default=128)
    parser.add_argument("--mode", choices=("flat", "direct", "both"), default="flat")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    worlds = [int(raw.strip()) for raw in args.worlds.split(",") if raw.strip()]
    print(f"device={wp.get_device()} block_dim={args.block_dim} n_runs={args.n_runs} mode={args.mode}")
    for scene in args.scenes:
        for num_worlds in worlds:
            run_case(args, scene, num_worlds)


if __name__ == "__main__":
    main()
