# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sweep PhoenX prepare-refresh stride on substepped rigid scenes.

prepare_refresh_stride freezes prepared per-row data for several substeps and
uses cached-prepare bookkeeping in between. The optimization is intended for
small substep timesteps, where Jacobians/effective masses change slowly. This
benchmark keeps the timing at full-frame level so it captures the actual
substep cadence used by robot and contact scenes.

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.bench_prepare_refresh_stride
"""

from __future__ import annotations

import argparse

import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _bench, _extract_solver
from newton._src.solvers.phoenx.benchmarks.scenarios import dr_legs, g1_flat, h1_flat, tower


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(raw.strip()) for raw in value.split(",") if raw.strip())


def _build_scene(
    scene: str,
    *,
    num_worlds: int,
    substeps: int,
    solver_iterations: int,
    prepare_refresh_stride: int,
):
    kwargs = {
        "num_worlds": num_worlds,
        "solver_name": "phoenx",
        "substeps": substeps,
        "solver_iterations": solver_iterations,
        "prepare_refresh_stride": prepare_refresh_stride,
    }
    if scene == "h1":
        return h1_flat.build(**kwargs)
    if scene == "g1":
        return g1_flat.build(**kwargs)
    if scene == "dr_legs":
        return dr_legs.build(**kwargs)
    if scene == "tower":
        return tower.build(**kwargs)
    raise ValueError(f"unknown scene: {scene}")


def _default_worlds(scene: str) -> int:
    if scene == "tower":
        return 32
    return 128


def _run_case(
    scene: str,
    num_worlds: int,
    stride: int,
    *,
    substeps: int,
    solver_iterations: int,
    prime_frames: int,
    warmup: int,
    n_runs: int,
    trials: int,
) -> tuple[float, float, int, int, int]:
    handle = _build_scene(
        scene,
        num_worlds=num_worlds,
        substeps=substeps,
        solver_iterations=solver_iterations,
        prepare_refresh_stride=stride,
    )
    solver = _extract_solver(handle)
    world = solver.world
    for _ in range(prime_frames):
        handle.simulate_one_frame()
    wp.synchronize_device()

    min_ms, med_ms = _bench(handle.simulate_one_frame, n_runs=n_runs, warmup=warmup, trials=trials)
    return min_ms, med_ms, int(world.prepare_refresh_stride), int(world.num_joints), int(world.max_contact_columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=("h1", "g1", "dr_legs", "tower"),
        default=["h1", "g1", "dr_legs"],
    )
    parser.add_argument("--worlds", type=_parse_csv_ints, default=())
    parser.add_argument("--strides", type=_parse_csv_ints, default=(1, 2, 3))
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--prime-frames", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--n-runs", type=int, default=8)
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()

    wp.init()
    print(
        f"device={wp.get_device()} substeps={args.substeps} strides={args.strides} "
        f"n_runs={args.n_runs} trials={args.trials}"
    )
    print("scene worlds stride chosen min_ms med_ms us_per_frame joints contacts rel_stride1 rel_best")

    explicit_worlds = tuple(args.worlds)
    for scene in args.scenes:
        world_counts = explicit_worlds or (_default_worlds(scene),)
        for num_worlds in world_counts:
            rows = []
            for stride in args.strides:
                min_ms, med_ms, chosen, joints, contacts = _run_case(
                    scene,
                    int(num_worlds),
                    int(stride),
                    substeps=args.substeps,
                    solver_iterations=args.solver_iterations,
                    prime_frames=args.prime_frames,
                    warmup=args.warmup,
                    n_runs=args.n_runs,
                    trials=args.trials,
                )
                rows.append((int(stride), chosen, min_ms, med_ms, joints, contacts))
            best = min(row[2] for row in rows)
            stride1 = next((row[2] for row in rows if row[0] == 1), rows[0][2])
            for stride, chosen, min_ms, med_ms, joints, contacts in rows:
                rel_stride1 = stride1 / min_ms if min_ms > 0.0 else float("nan")
                rel_best = min_ms / best if best > 0.0 else float("nan")
                print(
                    f"{scene:7s} {int(num_worlds):6d} {stride:6d} {chosen:6d} {min_ms:8.3f} {med_ms:8.3f} "
                    f"{1000.0 * min_ms / args.n_runs:12.3f} {joints:6d} {contacts:8d} "
                    f"{rel_stride1:11.3f} {rel_best:8.3f}"
                )


if __name__ == "__main__":
    main()
