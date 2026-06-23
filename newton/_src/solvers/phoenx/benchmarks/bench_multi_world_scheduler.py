# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Full-frame PhoenX multi-world scheduler tournament.

Compares production auto against explicit graph-capture-safe scheduler
shapes: fast-tail threads_per_world / worlds_per_block variants and
the private block-per-world scheduler candidates. Every candidate is measured
by building a fresh solver, resolving a fixed scheduler before capture, then
replaying a captured simulate_one_frame graph. No candidate switches
scheduler inside a captured graph.

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.bench_multi_world_scheduler --mode adaptive --scenes h1 g1 dr_legs tower --worlds 64,512
"""

from __future__ import annotations

import argparse
import time
import types
from collections.abc import Callable

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _extract_solver, _force_tpw
from newton._src.solvers.phoenx.benchmarks.scenarios import anymal_rl, dr_legs, g1_flat, h1_flat, tower
from newton._src.solvers.phoenx.solver import SolverPhoenX


def _build_scene(
    scene: str,
    num_worlds: int,
    *,
    substeps: int,
    solver_iterations: int,
    prepare_refresh_stride: int | str,
):
    if scene == "h1":
        return h1_flat.build(
            num_worlds, "phoenx", substeps, solver_iterations, prepare_refresh_stride=prepare_refresh_stride
        )
    if scene == "g1":
        return g1_flat.build(
            num_worlds, "phoenx", substeps, solver_iterations, prepare_refresh_stride=prepare_refresh_stride
        )
    if scene == "dr_legs":
        return dr_legs.build(
            num_worlds, "phoenx", substeps, solver_iterations, prepare_refresh_stride=prepare_refresh_stride
        )
    if scene == "tower":
        return tower.build(
            num_worlds,
            "phoenx",
            substeps,
            solver_iterations,
            step_layout="multi_world",
            prepare_refresh_stride=prepare_refresh_stride,
        )
    if scene == "anymal":
        return anymal_rl.build(
            num_worlds,
            "phoenx",
            substeps,
            solver_iterations,
            prepare_refresh_stride=prepare_refresh_stride,
        )
    raise ValueError(f"unknown scene {scene!r}")


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(raw.strip()) for raw in value.split(",") if raw.strip())


def _parse_stride_value(value: str) -> int | str:
    item = value.strip().lower()
    return "auto" if item == "auto" else int(item)


def _parse_tpw_values(value: str) -> tuple[int | str, ...]:
    out: list[int | str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        out.append("auto" if item == "auto" else int(item))
    return tuple(out)


def _format_tpw(tpw: int | str) -> str:
    return "auto" if tpw == "auto" else str(int(tpw))


def _scheduler_labels(args: argparse.Namespace) -> tuple[str, ...]:
    labels: list[str] = []
    if args.include_auto:
        labels.append("auto")
    if args.include_fast_tail_default:
        labels.append("fast_tail")

    for tpw in args.fast_tail_tpw:
        for wpb in args.fast_tail_wpb:
            labels.append(f"fast_tail_tpw{_format_tpw(tpw)}_wpb{int(wpb)}")

    labels.extend(f"block_world_{dim}" for dim in args.block_world_dims)

    # Keep first occurrence so users may include an explicit candidate that
    # matches the default without measuring it twice.
    unique: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            unique.append(label)
            seen.add(label)
    return tuple(unique)


def _set_worlds_per_block(world, worlds_per_block: int) -> None:
    world._fast_tail_worlds_per_block = types.MethodType(
        lambda self, _worlds_per_block=int(worlds_per_block): _worlds_per_block,
        world,
    )


def _apply_scheduler(solver: SolverPhoenX, label: str) -> None:
    world = solver.world
    if label == "auto":
        world._configure_multi_world_scheduler("auto")
        return
    if label == "fast_tail":
        world._configure_multi_world_scheduler("fast_tail")
        return

    prefix = "fast_tail_tpw"
    if label.startswith(prefix):
        rest = label[len(prefix) :]
        tpw_raw, wpb_raw = rest.split("_wpb", 1)
        world._configure_multi_world_scheduler("fast_tail")
        _force_tpw(solver, "auto" if tpw_raw == "auto" else int(tpw_raw))
        _set_worlds_per_block(world, int(wpb_raw))
        return

    prefix = "block_world_"
    if label.startswith(prefix):
        world._multi_world_scheduler = "block_world"
        world._multi_world_block_dim = int(label[len(prefix) :])
        return

    raise ValueError(f"unknown scheduler candidate {label!r}")


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


def _measure_scheduler(
    args: argparse.Namespace,
    scene: str,
    num_worlds: int,
    scheduler: str,
    *,
    n_runs: int,
    warmup: int,
    prime_frames: int,
) -> tuple[str, float, float, str] | None:
    handle = _build_scene(
        scene,
        num_worlds,
        substeps=args.substeps,
        solver_iterations=args.solver_iterations,
        prepare_refresh_stride=args.prepare_refresh_stride,
    )
    solver = _extract_solver(handle)
    _apply_scheduler(solver, scheduler)
    if scheduler.startswith("block_world") and not solver.world._block_world_supported():
        return None

    for _ in range(prime_frames):
        handle.simulate_one_frame()
    wp.synchronize_device()

    world = solver.world
    resolved = (
        f"kind={world._multi_world_scheduler}"
        f",tpw={int(world._tpw_choice.numpy()[0])}"
        f",wpb={int(world._fast_tail_worlds_per_block())}"
        f",block_dim={int(world._multi_world_block_dim)}"
        f",family={bool(world._fast_tail_family_split())}"
        f",prep={int(world.prepare_refresh_stride)}"
    )
    min_ms, med_ms = _bench(handle.simulate_one_frame, n_runs=n_runs, warmup=warmup, trials=args.trials)
    return scheduler, min_ms, med_ms, resolved


def _baseline_label(results: list[tuple[str, float, float, str]]) -> str:
    labels = {label for label, _min_ms, _med_ms, _resolved in results}
    if "auto" in labels:
        return "auto"
    if "fast_tail" in labels:
        return "fast_tail"
    return results[0][0]


def _format_results(results: list[tuple[str, float, float, str]]) -> tuple[str, float, str]:
    baseline_name = _baseline_label(results)
    baseline = next(min_ms for label, min_ms, _med_ms, _resolved in results if label == baseline_name)
    pieces = []
    for label, min_ms, _med_ms, resolved in results:
        speed = baseline / min_ms if min_ms > 0.0 else float("nan")
        pieces.append(f"{label}={min_ms:8.3f}ms({speed:5.3f}x,{resolved})")
    best_label, best_min, _best_med, _best_resolved = min(results, key=lambda item: item[1])
    best_speed = baseline / best_min if best_min > 0.0 else float("nan")
    return best_label, best_speed, " ".join(pieces)


def _run_sweep_case(args: argparse.Namespace, scene: str, num_worlds: int) -> None:
    results: list[tuple[str, float, float, str]] = []
    for scheduler in _scheduler_labels(args):
        measured = _measure_scheduler(
            args,
            scene,
            num_worlds,
            scheduler,
            n_runs=args.n_runs,
            warmup=args.warmup,
            prime_frames=args.prime_frames,
        )
        if measured is None:
            print(f"{scene:7s} worlds={num_worlds:5d} scheduler={scheduler} unsupported")
            continue
        results.append(measured)

    best_label, best_speed, pieces = _format_results(results)
    print(f"{scene:7s} worlds={num_worlds:5d} best={best_label} best_speedup={best_speed:5.3f}x " + pieces)


def _run_adaptive_case(args: argparse.Namespace, scene: str, num_worlds: int) -> None:
    tournament: list[tuple[str, float, float, str]] = []
    for scheduler in _scheduler_labels(args):
        measured = _measure_scheduler(
            args,
            scene,
            num_worlds,
            scheduler,
            n_runs=args.adapt_runs,
            warmup=args.adapt_warmup,
            prime_frames=args.prime_frames,
        )
        if measured is not None:
            tournament.append(measured)

    best_label, predicted_speed, tournament_pieces = _format_results(tournament)
    selected = best_label if predicted_speed >= args.adapt_min_speedup else "auto"

    verify_labels = ["auto"] if selected == "auto" else ["auto", selected]
    verified: list[tuple[str, float, float, str]] = []
    for scheduler in verify_labels:
        measured = _measure_scheduler(
            args,
            scene,
            num_worlds,
            scheduler,
            n_runs=args.n_runs,
            warmup=args.warmup,
            prime_frames=args.prime_frames,
        )
        if measured is not None:
            verified.append(measured)
    _best_verified, verified_speed, verified_pieces = _format_results(verified)
    accepted = selected if selected != "auto" and verified_speed >= args.adapt_min_speedup else "auto"
    print(
        f"{scene:7s} worlds={num_worlds:5d} mode=adaptive candidate={selected} accepted={accepted} "
        f"predicted={predicted_speed:5.3f}x verified={verified_speed:5.3f}x "
        f"tournament=[{tournament_pieces}] verify=[{verified_pieces}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=("h1", "g1", "dr_legs", "tower", "anymal"),
        default=["h1", "g1", "dr_legs", "tower"],
    )
    parser.add_argument("--mode", choices=("sweep", "adaptive"), default="sweep")
    parser.add_argument("--worlds", type=_parse_csv_ints, default=(64, 512))
    parser.add_argument("--include-auto", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-fast-tail-default", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-tail-tpw", type=_parse_tpw_values, default=(32, 16))
    parser.add_argument("--fast-tail-wpb", type=_parse_csv_ints, default=(1, 2, 4))
    parser.add_argument("--block-world-dims", type=_parse_csv_ints, default=(32, 64, 128))
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--prepare-refresh-stride", type=_parse_stride_value, default="auto")
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--prime-frames", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=32)
    parser.add_argument("--adapt-runs", type=int, default=8)
    parser.add_argument("--adapt-warmup", type=int, default=1)
    parser.add_argument("--adapt-min-speedup", type=float, default=1.03)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()

    wp.init()
    print(
        f"device={wp.get_device()} mode={args.mode} n_runs={args.n_runs} adapt_runs={args.adapt_runs} "
        f"prepare_refresh_stride={args.prepare_refresh_stride} "
        f"fast_tail_tpw={args.fast_tail_tpw} fast_tail_wpb={args.fast_tail_wpb} "
        f"block_world_dims={args.block_world_dims}"
    )
    for scene in args.scenes:
        for num_worlds in args.worlds:
            if args.mode == "adaptive":
                _run_adaptive_case(args, scene, num_worlds)
            else:
                _run_sweep_case(args, scene, num_worlds)


if __name__ == "__main__":
    main()
