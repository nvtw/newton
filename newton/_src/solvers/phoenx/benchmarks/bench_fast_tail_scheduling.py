# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sweep PhoenX multi-world fast-tail scheduling choices.

This benchmark complements ``bench_threads_per_world`` by sweeping both
``threads_per_world`` and fast-tail ``worlds_per_block`` packing. It is
research tooling: production still chooses these values heuristically, while
this script measures the sensitivity on representative scenes before we tune
those heuristics.

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.bench_fast_tail_scheduling
"""

from __future__ import annotations

import argparse
import types

import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _bench, _extract_solver, _force_tpw
from newton._src.solvers.phoenx.benchmarks.scenarios import g1_flat, h1_flat, tower


def _build_scene(scene: str, num_worlds: int, *, substeps: int, solver_iterations: int):
    if scene == "h1":
        return h1_flat.build(num_worlds, "phoenx", substeps, solver_iterations)
    if scene == "g1":
        return g1_flat.build(num_worlds, "phoenx", substeps, solver_iterations)
    if scene == "tower":
        return tower.build(
            num_worlds,
            "phoenx",
            substeps,
            solver_iterations,
            step_layout="multi_world",
        )
    raise ValueError(f"unknown scene: {scene}")


def _set_worlds_per_block(world, worlds_per_block: int | None) -> None:
    if worlds_per_block is None:
        return
    world._fast_tail_worlds_per_block = types.MethodType(
        lambda self, _worlds_per_block=worlds_per_block: _worlds_per_block,
        world,
    )


def _measure_case(
    *,
    scene: str,
    num_worlds: int,
    tpw: int | str,
    worlds_per_block: int | None,
    substeps: int,
    solver_iterations: int,
    warmup: int,
    n_runs: int,
    trials: int,
) -> tuple[float, float, int, int, int, int, float]:
    handle = _build_scene(scene, num_worlds, substeps=substeps, solver_iterations=solver_iterations)
    solver = _extract_solver(handle)
    world = solver.world
    _force_tpw(solver, tpw)
    _set_worlds_per_block(world, worlds_per_block)

    for _ in range(warmup):
        handle.simulate_one_frame()
    wp.synchronize_device()

    chosen_tpw = int(world._tpw_choice.numpy()[0]) if tpw == "auto" else int(tpw)
    chosen_wpb = int(world._fast_tail_worlds_per_block())
    num_colors = int(world._world_num_colors.numpy().max(initial=0))
    total_cids = int(world._world_csr_offsets.numpy()[world.num_worlds])
    contacts_per_world = float(world.max_contact_columns) / float(max(1, world.num_worlds))
    min_ms, med_ms = _bench(handle.simulate_one_frame, n_runs=n_runs, warmup=0, trials=trials)
    return min_ms, med_ms, chosen_tpw, chosen_wpb, num_colors, total_cids, contacts_per_world


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _parse_tpw_values(value: str) -> tuple[int | str, ...]:
    out: list[int | str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        out.append("auto" if item == "auto" else int(item))
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes", nargs="+", choices=("h1", "g1", "tower"), default=["h1", "tower"])
    parser.add_argument("--worlds", type=_parse_csv_ints, default=(32, 64, 128, 512))
    parser.add_argument("--tpw", type=_parse_tpw_values, default=("auto", 32, 16, 8))
    parser.add_argument("--wpb", type=_parse_csv_ints, default=(1, 2, 4, 8))
    parser.add_argument("--include-default", action="store_true", help="Also measure production heuristic wpb.")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=40)
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()

    wp.init()
    print(f"device={wp.get_device()} sm_count={getattr(wp.get_device(), 'sm_count', 'N/A')}")
    print(
        "scene worlds tpw wpb chosen_tpw chosen_wpb min_ms med_ms us_per_frame "
        "max_colors total_cids contacts_per_world rel_best"
    )

    for scene in args.scenes:
        for num_worlds in args.worlds:
            rows: list[tuple] = []
            wpb_values: tuple[int | None, ...]
            if args.include_default:
                wpb_values = (None, *args.wpb)
            else:
                wpb_values = args.wpb
            for tpw in args.tpw:
                for worlds_per_block in wpb_values:
                    min_ms, med_ms, chosen_tpw, chosen_wpb, num_colors, total_cids, contacts_per_world = _measure_case(
                        scene=scene,
                        num_worlds=num_worlds,
                        tpw=tpw,
                        worlds_per_block=worlds_per_block,
                        substeps=args.substeps,
                        solver_iterations=args.solver_iterations,
                        warmup=args.warmup,
                        n_runs=args.n_runs,
                        trials=args.trials,
                    )
                    rows.append(
                        (
                            min_ms,
                            med_ms,
                            scene,
                            num_worlds,
                            tpw,
                            "default" if worlds_per_block is None else worlds_per_block,
                            chosen_tpw,
                            chosen_wpb,
                            num_colors,
                            total_cids,
                            contacts_per_world,
                        )
                    )
            best = min(row[0] for row in rows)
            for row in sorted(rows, key=lambda r: (str(r[4]), str(r[5]))):
                (
                    min_ms,
                    med_ms,
                    row_scene,
                    row_num_worlds,
                    row_tpw,
                    wpb,
                    chosen_tpw,
                    chosen_wpb,
                    num_colors,
                    total_cids,
                    contacts_per_world,
                ) = row
                rel_best = min_ms / best if best > 0.0 else float("nan")
                print(
                    f"{row_scene:5s} {row_num_worlds:6d} {row_tpw!s:>4s} {wpb!s:>7s} "
                    f"{chosen_tpw:10d} {chosen_wpb:10d} {min_ms:8.3f} {med_ms:8.3f} "
                    f"{1000.0 * min_ms / args.n_runs:12.3f} {num_colors:10d} {total_cids:10d} "
                    f"{contacts_per_world:18.1f} {rel_best:8.3f}"
                )


if __name__ == "__main__":
    main()
