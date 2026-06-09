# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Full-frame PhoenX multi-world scheduler benchmark.

Compares the production fast-tail dispatcher with the private block-per-world
scheduler candidate. Unlike the isolated color-grid scheduler benches, this
captures an entire ``simulate_one_frame`` call, so the number includes contact
ingest, coloring, solve, relax, integration, and graph replay overhead.

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.bench_multi_world_scheduler \
        --scenes h1 g1 dr_legs tower --worlds 64,512
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _extract_solver
from newton._src.solvers.phoenx.benchmarks.scenarios import dr_legs, g1_flat, h1_flat, tower
from newton._src.solvers.phoenx.solver import SolverPhoenX


def _build_scene(scene: str, num_worlds: int, *, substeps: int, solver_iterations: int):
    if scene == "h1":
        return h1_flat.build(num_worlds, "phoenx", substeps, solver_iterations)
    if scene == "g1":
        return g1_flat.build(num_worlds, "phoenx", substeps, solver_iterations)
    if scene == "dr_legs":
        return dr_legs.build(num_worlds, "phoenx", substeps, solver_iterations)
    if scene == "tower":
        return tower.build(num_worlds, "phoenx", substeps, solver_iterations, step_layout="multi_world")
    raise ValueError(f"unknown scene {scene!r}")


def _parse_worlds(value: str) -> tuple[int, ...]:
    return tuple(int(raw.strip()) for raw in value.split(",") if raw.strip())


def _scheduler_labels(block_dims: str) -> tuple[str, ...]:
    labels = ["fast_tail"]
    labels.extend(f"block_world_{dim}" for dim in _parse_worlds(block_dims))
    return tuple(labels)


def _apply_scheduler(solver: SolverPhoenX, label: str) -> None:
    world = solver.world
    if label == "fast_tail":
        world._multi_world_scheduler = "fast_tail"
        return
    prefix = "block_world_"
    if not label.startswith(prefix):
        raise ValueError(f"unknown scheduler {label!r}")
    world._multi_world_scheduler = "block_world"
    world._multi_world_block_dim = int(label[len(prefix) :])


def _bench(simulate: Callable[[], None], *, n_runs: int, warmup: int, trials: int) -> tuple[float, float]:
    for _ in range(warmup):
        simulate()
    wp.synchronize_device()

    with wp.ScopedCapture() as capture:
        simulate()
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


def _run_case(args: argparse.Namespace, scene: str, num_worlds: int) -> None:
    results: list[tuple[str, float, float]] = []
    for scheduler in _scheduler_labels(args.block_world_dims):
        handle = _build_scene(
            scene,
            num_worlds,
            substeps=args.substeps,
            solver_iterations=args.solver_iterations,
        )
        solver = _extract_solver(handle)
        _apply_scheduler(solver, scheduler)
        if scheduler != "fast_tail" and not solver.world._block_world_supported():
            print(f"{scene:7s} worlds={num_worlds:5d} scheduler={scheduler} unsupported")
            continue

        # Prime with the selected scheduler so contacts, coloring, and kernel
        # modules are warm before capture.
        for _ in range(args.prime_frames):
            handle.simulate_one_frame()
        wp.synchronize_device()

        min_ms, med_ms = _bench(handle.simulate_one_frame, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials)
        results.append((scheduler, min_ms, med_ms))

    baseline = next(min_ms for label, min_ms, _med_ms in results if label == "fast_tail")
    pieces = []
    for label, min_ms, _med_ms in results:
        speed = baseline / min_ms if min_ms > 0.0 else float("nan")
        pieces.append(f"{label}={min_ms:8.3f}ms({speed:5.3f}x)")
    best_label, best_min, _best_med = min(results, key=lambda item: item[1])
    best_speed = baseline / best_min if best_min > 0.0 else float("nan")
    print(
        f"{scene:7s} worlds={num_worlds:5d} best={best_label} best_speedup={best_speed:5.3f}x "
        + " ".join(pieces)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes", nargs="+", choices=("h1", "g1", "dr_legs", "tower"), default=["h1", "g1", "dr_legs", "tower"])
    parser.add_argument("--worlds", default="64,512")
    parser.add_argument("--block-world-dims", default="32,64,128")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--prime-frames", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=32)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()

    wp.init()
    print(f"device={wp.get_device()} n_runs={args.n_runs} block_world_dims={args.block_world_dims}")
    for scene in args.scenes:
        for num_worlds in _parse_worlds(args.worlds):
            _run_case(args, scene, num_worlds)


if __name__ == "__main__":
    main()
