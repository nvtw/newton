# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sweep the PhoenX single-world fused-tail threshold.

The single-world solver launches a persistent head kernel until remaining color
classes are small enough, then drains those tail colors in one block with
__syncthreads between colors. This benchmark measures that threshold on
robot scenes as well as a contact-stack guardrail so tuning does not overfit to
contact-only workloads.

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.bench_singleworld_tail_fuse
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _extract_solver
from newton._src.solvers.phoenx.benchmarks.scenarios import dr_legs, g1_flat, h1_flat, tower, tower_grid
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(raw.strip()) for raw in value.split(",") if raw.strip())


def _build_scene(
    scene: str,
    *,
    substeps: int,
    solver_iterations: int,
    tower_grid_side: int,
    tower_grid_layers: int,
):
    kwargs = {
        "num_worlds": 1,
        "solver_name": "phoenx",
        "substeps": substeps,
        "solver_iterations": solver_iterations,
        "step_layout": "single_world",
    }
    if scene == "h1":
        return h1_flat.build(**kwargs)
    if scene == "g1":
        return g1_flat.build(**kwargs)
    if scene == "dr_legs":
        return dr_legs.build(**kwargs)
    if scene == "tower":
        return tower.build(**kwargs)
    if scene == "tower_grid":
        return tower_grid.build(**kwargs, grid_side=tower_grid_side, layers=tower_grid_layers)
    raise ValueError(f"unknown scene: {scene}")


def _set_tail_config(world: PhoenXWorld, *, threshold: int, block_dim: int) -> None:
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if threshold > block_dim:
        raise ValueError(f"threshold {threshold} must be <= block_dim {block_dim}")
    world._fuse_threshold = int(threshold)
    world._fuse_tail_block_dim = int(block_dim)


def _tail_stats(world: PhoenXWorld, threshold: int) -> tuple[int, int, int, int]:
    starts = world._partitioner.color_starts.numpy()
    num_colors = int(world._partitioner.num_colors.numpy()[0])
    if num_colors <= 0:
        return 0, 0, 0, 0
    counts = np.diff(starts[: num_colors + 1]).astype(np.int64, copy=False)
    max_color = int(counts.max(initial=0))
    tail_mask = counts <= int(threshold) if threshold > 0 else np.zeros_like(counts, dtype=bool)
    return num_colors, max_color, int(tail_mask.sum()), int(counts[tail_mask].sum())


def _bench_solve(world: PhoenXWorld, *, n_runs: int, warmup: int, trials: int) -> tuple[float, float]:
    solve = world._solve_main_singleworld
    for _ in range(warmup):
        solve()
    wp.synchronize_device()

    with wp.ScopedCapture(device=world.device) as capture:
        solve()
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


def _run_case(
    scene: str,
    threshold: int,
    *,
    block_dim: int,
    substeps: int,
    solver_iterations: int,
    tower_grid_side: int,
    tower_grid_layers: int,
    prime_frames: int,
    warmup: int,
    n_runs: int,
    trials: int,
) -> tuple[float, float, int, int, int, int]:
    handle = _build_scene(
        scene,
        substeps=substeps,
        solver_iterations=solver_iterations,
        tower_grid_side=tower_grid_side,
        tower_grid_layers=tower_grid_layers,
    )
    solver = _extract_solver(handle)
    world = solver.world
    if world.step_layout != "single_world":
        raise RuntimeError(f"expected single_world, got {world.step_layout!r}")
    _set_tail_config(world, threshold=threshold, block_dim=block_dim)

    for _ in range(prime_frames):
        handle.simulate_one_frame()
    wp.synchronize_device()

    num_colors, max_color, tail_colors, tail_rows = _tail_stats(world, threshold)
    min_ms, med_ms = _bench_solve(world, n_runs=n_runs, warmup=warmup, trials=trials)
    return min_ms, med_ms, num_colors, max_color, tail_colors, tail_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=("h1", "g1", "dr_legs", "tower", "tower_grid"),
        default=["h1", "g1", "dr_legs", "tower"],
    )
    parser.add_argument("--thresholds", type=_parse_csv_ints, default=(0, 64, 128, 256))
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--tower-grid-side", type=int, default=2)
    parser.add_argument("--tower-grid-layers", type=int, default=16)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--prime-frames", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()

    wp.init()
    print(
        f"device={wp.get_device()} thresholds={args.thresholds} block_dim={args.block_dim} "
        f"tower_grid={args.tower_grid_side}x{args.tower_grid_side}x{args.tower_grid_layers} "
        f"n_runs={args.n_runs} trials={args.trials}"
    )
    print("scene threshold min_ms med_ms us_per_solve colors max_color tail_colors tail_rows rel_best")

    for scene in args.scenes:
        rows = []
        for threshold in args.thresholds:
            min_ms, med_ms, num_colors, max_color, tail_colors, tail_rows = _run_case(
                scene,
                int(threshold),
                block_dim=args.block_dim,
                substeps=args.substeps,
                solver_iterations=args.solver_iterations,
                tower_grid_side=args.tower_grid_side,
                tower_grid_layers=args.tower_grid_layers,
                prime_frames=args.prime_frames,
                warmup=args.warmup,
                n_runs=args.n_runs,
                trials=args.trials,
            )
            rows.append((int(threshold), min_ms, med_ms, num_colors, max_color, tail_colors, tail_rows))
        best = min(row[1] for row in rows)
        for threshold, min_ms, med_ms, num_colors, max_color, tail_colors, tail_rows in rows:
            rel_best = min_ms / best if best > 0.0 else float("nan")
            print(
                f"{scene:7s} {threshold:9d} {min_ms:8.3f} {med_ms:8.3f} "
                f"{1000.0 * min_ms / args.n_runs:12.3f} {num_colors:6d} {max_color:9d} "
                f"{tail_colors:11d} {tail_rows:9d} {rel_best:8.3f}"
            )


if __name__ == "__main__":
    main()
