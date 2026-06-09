# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Experimental PhoenX dependency-wave solve benchmark.
# CPU builds an ASAP schedule from colored rows and touched body nodes.
# The flat mode runs the resulting waves as captured per-wave kernels;
# mega mode keeps the older persistent-kernel software barrier path.
# Rows that touch the same dynamic body keep PGS order; independent
# later-color rows can move into earlier waves when the graph permits it.

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _extract_solver
from newton._src.solvers.phoenx.benchmarks.experimental.bench_color_grid_actual_solve import (
    ColorGridHost,
    _bench,
    _build_scene,
    _color_grid_mega_runner,
    _color_grid_runner,
    _extract_color_grid,
    _upload_color_grid,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

MAX_NODES_PER_ROW = 8


@dataclass
class WaveStats:
    rows: int
    colors: int
    waves: int
    max_color_rows: int
    max_wave_rows: int
    mean_wave_rows: float


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(raw.strip()) for raw in value.split(",") if raw.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(raw.strip()) for raw in value.split(",") if raw.strip())


def _tail_fractions(args: argparse.Namespace) -> tuple[float, ...]:
    raw_values = _parse_csv_floats(args.tail_fractions) if args.tail_fractions.strip() else (float(args.tail_fraction),)
    values: list[float] = []
    seen: set[int] = set()
    for raw in raw_values:
        value = max(0.0, min(0.95, float(raw)))
        key = int(round(value * 1.0e6))
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return tuple(values) if values else (0.0,)


def _format_fraction(value: float) -> str:
    return f"{value:g}"


def _unique_row_nodes(elements: np.ndarray, eid: int) -> list[int]:
    nodes: list[int] = []
    seen: set[int] = set()
    for raw_node in elements[eid]["bodies"]:
        node = int(raw_node)
        if node < 0 or node in seen:
            continue
        seen.add(node)
        nodes.append(node)
        if len(nodes) >= MAX_NODES_PER_ROW:
            break
    return nodes


def _extract_dependency_wave_grid(world: PhoenXWorld) -> tuple[ColorGridHost, WaveStats]:
    active = int(world._num_active_constraints.numpy()[0])
    elements = world._elements.numpy()
    eids_by_color = world._world_element_ids_by_color.numpy()
    starts = world._world_color_starts.numpy()
    csr = world._world_csr_offsets.numpy()
    num_colors_per_world = world._world_num_colors.numpy().astype(np.int32, copy=False)

    waves: list[list[int]] = []
    last_wave_for_node: dict[int, int] = {}

    def append_to_wave(wave_id: int, eid: int, nodes: list[int]) -> None:
        while len(waves) <= wave_id:
            waves.append([])
        waves[wave_id].append(eid)
        for node in nodes:
            last_wave_for_node[node] = wave_id

    for world_id in range(world.num_worlds):
        world_base = int(csr[world_id])
        for color in range(int(num_colors_per_world[world_id])):
            start = world_base + int(starts[world_id, color])
            end = world_base + int(starts[world_id, color + 1])
            for cursor in range(start, end):
                eid = int(eids_by_color[cursor])
                if eid < 0 or eid >= active:
                    continue
                nodes = _unique_row_nodes(elements, eid)
                wave_id = 0
                for node in nodes:
                    wave_id = max(wave_id, last_wave_for_node.get(node, -1) + 1)
                append_to_wave(wave_id, eid, nodes)

    if not waves:
        waves = [[0]]

    wave_counts = np.asarray([len(w) for w in waves], dtype=np.int32)
    eids: list[int] = []
    wave_starts = np.zeros(len(waves) + 1, dtype=np.int32)
    for wave_id, wave_rows in enumerate(waves):
        wave_starts[wave_id] = len(eids)
        eids.extend(wave_rows)
    wave_starts[len(waves)] = len(eids)
    if not eids:
        eids.append(0)

    color_grid = _extract_color_grid(world)
    color_counts = np.diff(color_grid.color_starts[: color_grid.num_colors + 1])
    stats = WaveStats(
        rows=len(eids),
        colors=int(color_grid.num_colors),
        waves=len(waves),
        max_color_rows=int(color_counts.max(initial=0)),
        max_wave_rows=int(wave_counts.max(initial=0)),
        mean_wave_rows=float(wave_counts.mean()) if wave_counts.size else 0.0,
    )
    return (
        ColorGridHost(
            eids=np.asarray(eids, dtype=np.int32),
            color_starts=wave_starts,
            color_max_counts=wave_counts,
            num_colors=len(waves),
        ),
        stats,
    )


def _append_scheduled_row(
    waves: list[list[int]],
    last_wave_for_node: dict[int, int],
    eid: int,
    nodes: list[int],
    release_wave: int,
) -> None:
    wave_id = int(release_wave)
    for node in nodes:
        wave_id = max(wave_id, last_wave_for_node.get(node, -1) + 1)
    while len(waves) <= wave_id:
        waves.append([])
    waves[wave_id].append(eid)
    for node in nodes:
        last_wave_for_node[node] = wave_id


def _color_rows_with_nodes(world: PhoenXWorld) -> tuple[list[list[tuple[int, list[int]]]], ColorGridHost]:
    active = int(world._num_active_constraints.numpy()[0])
    elements = world._elements.numpy()
    color_grid = _extract_color_grid(world)
    rows_by_color: list[list[tuple[int, list[int]]]] = []
    for color in range(color_grid.num_colors):
        start = int(color_grid.color_starts[color])
        end = int(color_grid.color_starts[color + 1])
        color_rows: list[tuple[int, list[int]]] = []
        for cursor in range(start, end):
            eid = int(color_grid.eids[cursor])
            if eid < 0 or eid >= active:
                continue
            color_rows.append((eid, _unique_row_nodes(elements, eid)))
        rows_by_color.append(color_rows)
    return rows_by_color, color_grid


def _make_wave_host(waves: list[list[int]], color_grid: ColorGridHost) -> tuple[ColorGridHost, WaveStats]:
    if not waves:
        waves = [[0]]
    wave_counts = np.asarray([len(w) for w in waves], dtype=np.int32)
    eids: list[int] = []
    wave_starts = np.zeros(len(waves) + 1, dtype=np.int32)
    for wave_id, wave_rows in enumerate(waves):
        wave_starts[wave_id] = len(eids)
        eids.extend(wave_rows)
    wave_starts[len(waves)] = len(eids)
    if not eids:
        eids.append(0)
    color_counts = np.diff(color_grid.color_starts[: color_grid.num_colors + 1])
    stats = WaveStats(
        rows=len(eids),
        colors=int(color_grid.num_colors),
        waves=len(waves),
        max_color_rows=int(color_counts.max(initial=0)),
        max_wave_rows=int(wave_counts.max(initial=0)),
        mean_wave_rows=float(wave_counts.mean()) if wave_counts.size else 0.0,
    )
    return (
        ColorGridHost(
            eids=np.asarray(eids, dtype=np.int32),
            color_starts=wave_starts,
            color_max_counts=wave_counts,
            num_colors=len(waves),
        ),
        stats,
    )


def _extract_tail_overlap_wave_grid(world: PhoenXWorld, tail_fraction: float) -> tuple[ColorGridHost, WaveStats]:
    rows_by_color, color_grid = _color_rows_with_nodes(world)
    waves: list[list[int]] = []
    last_wave_for_node: dict[int, int] = {}
    frac = max(0.0, min(0.95, float(tail_fraction)))

    for color, rows in enumerate(rows_by_color):
        if not rows:
            continue
        tail_count = int(np.ceil(float(len(rows)) * frac)) if frac > 0.0 else 0
        tail_count = min(max(tail_count, 0), max(0, len(rows) - 1))
        if tail_count > 0:
            order = sorted(
                range(len(rows)),
                key=lambda i: (1 if rows[i][0] >= world.num_joints else 0, len(rows[i][1]), i),
            )
            tail_indices = set(order[-tail_count:])
        else:
            tail_indices = set()

        for i, (eid, nodes) in enumerate(rows):
            if i not in tail_indices:
                _append_scheduled_row(waves, last_wave_for_node, eid, nodes, color)
        for i, (eid, nodes) in enumerate(rows):
            if i in tail_indices:
                _append_scheduled_row(waves, last_wave_for_node, eid, nodes, color + 1)

    return _make_wave_host(waves, color_grid)


def _run_case(args: argparse.Namespace, scene: str, num_worlds: int) -> None:
    handle = _build_scene(scene, num_worlds, substeps=args.substeps, solver_iterations=args.solver_iterations)
    solver = _extract_solver(handle)
    world = solver.world
    if world.step_layout == "single_world" or world.mass_splitting_enabled or world.num_particles > 0:
        raise RuntimeError("wave scheduler prototype only supports multi_world rigid scenes")
    if world._has_soft_contact_pd or world._sleeping_enabled:
        raise RuntimeError("wave scheduler prototype excludes soft-contact PD and sleeping")

    for _ in range(args.prime_frames):
        handle.simulate_one_frame()
    wp.synchronize_device()

    color_host = _extract_color_grid(world)
    wave_host, stats = _extract_dependency_wave_grid(world)
    tail_cases = [(fraction, *_extract_tail_overlap_wave_grid(world, fraction)) for fraction in _tail_fractions(args)]
    color_graph = _upload_color_grid(color_host, world.device)
    wave_graph = _upload_color_grid(wave_host, world.device)
    tail_graphs = [
        (fraction, tail_stats, _upload_color_grid(tail_host, world.device))
        for fraction, tail_host, tail_stats in tail_cases
    ]

    def make_run(graph):
        if args.mode == "mega":
            return _color_grid_mega_runner(
                world,
                graph,
                block_dim=args.block_dim,
                worker_blocks=args.worker_blocks,
            )
        return _color_grid_runner(world, graph, block_dim=args.block_dim)

    color_run = make_run(color_graph)
    wave_run = make_run(wave_graph)
    tail_runs = [(fraction, tail_stats, make_run(tail_graph)) for fraction, tail_stats, tail_graph in tail_graphs]

    color_run()
    wave_run()
    for _fraction, _tail_stats, tail_run in tail_runs:
        tail_run()
    wp.synchronize_device()

    base_min, _base_med = _bench(
        world._solve_main,
        n_runs=args.n_runs,
        warmup=args.warmup,
        trials=args.trials,
        device=world.device,
    )
    color_min, _color_med = _bench(
        color_run,
        n_runs=args.n_runs,
        warmup=args.warmup,
        trials=args.trials,
        device=world.device,
    )
    wave_min, _wave_med = _bench(
        wave_run,
        n_runs=args.n_runs,
        warmup=args.warmup,
        trials=args.trials,
        device=world.device,
    )
    tail_results: list[tuple[float, WaveStats, float]] = []
    for fraction, tail_stats, tail_run in tail_runs:
        tail_min, _tail_med = _bench(
            tail_run,
            n_runs=args.n_runs,
            warmup=args.warmup,
            trials=args.trials,
            device=world.device,
        )
        tail_results.append((fraction, tail_stats, tail_min))

    color_speed = base_min / color_min if color_min > 0.0 else float("nan")
    wave_speed = base_min / wave_min if wave_min > 0.0 else float("nan")
    suffix = "mega" if args.mode == "mega" else "flat"
    candidates: list[tuple[str, float]] = [
        ("baseline", base_min),
        (f"color_{suffix}", color_min),
        (f"wave_{suffix}", wave_min),
    ]
    tail_pieces: list[str] = []
    tail_stat_pieces: list[str] = []
    for fraction, tail_stats, tail_min in tail_results:
        label = f"tail{_format_fraction(fraction)}_{suffix}"
        tail_speed = base_min / tail_min if tail_min > 0.0 else float("nan")
        candidates.append((label, tail_min))
        tail_pieces.append(f"{label}={tail_min:8.3f}ms({tail_speed:5.3f}x)")
        tail_stat_pieces.append(f"{_format_fraction(fraction)}:waves={tail_stats.waves},max={tail_stats.max_wave_rows}")
    best_label, best_min = min(candidates, key=lambda item: item[1])
    best_speed = base_min / best_min if best_min > 0.0 else float("nan")
    print(
        f"{scene:7s} worlds={num_worlds:5d} rows={stats.rows:7d} colors={stats.colors:4d} "
        f"waves={stats.waves:4d} max_color={stats.max_color_rows:5d} max_wave={stats.max_wave_rows:5d} "
        f"tails=[{';'.join(tail_stat_pieces)}] baseline={base_min:8.3f}ms "
        f"color_{suffix}={color_min:8.3f}ms({color_speed:5.3f}x) "
        f"wave_{suffix}={wave_min:8.3f}ms({wave_speed:5.3f}x) "
        + " ".join(tail_pieces)
        + f" auto_best={best_label}:{best_min:8.3f}ms({best_speed:5.3f}x) "
        f"base_us={1000.0 * base_min / args.n_runs:8.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental PhoenX dependency-wave solve benchmark")
    parser.add_argument(
        "--scenes", nargs="+", choices=("h1", "g1", "dr_legs", "tower"), default=["h1", "g1", "dr_legs", "tower"]
    )
    parser.add_argument("--worlds", type=_parse_csv_ints, default=(32,))
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--prime-frames", type=int, default=3)
    parser.add_argument("--mode", choices=("flat", "mega"), default="flat")
    parser.add_argument("--block-dim", type=int, default=128)
    parser.add_argument("--worker-blocks", type=int, default=128)
    parser.add_argument(
        "--tail-fractions",
        default="0,0.125,0.25,0.5",
        help="Comma-separated tail-overlap fractions to build and time for auto-best reporting.",
    )
    parser.add_argument("--tail-fraction", type=float, default=0.25, help="Fallback when --tail-fractions is empty.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--n-runs", type=int, default=8)
    parser.add_argument("--trials", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    print(
        f"device={wp.get_device()} mode={args.mode} block_dim={args.block_dim} worker_blocks={args.worker_blocks} "
        f"tail_fractions={_tail_fractions(args)} n_runs={args.n_runs} trials={args.trials}"
    )
    for scene in args.scenes:
        for num_worlds in args.worlds:
            _run_case(args, scene, int(num_worlds))


if __name__ == "__main__":
    main()
