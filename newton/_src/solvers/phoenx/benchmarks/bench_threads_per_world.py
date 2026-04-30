# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end micro-benchmark for the adaptive ``threads_per_world`` knob.

The PhoenX ``multi_world`` fast-tail kernels read their effective
threads-per-world from a 1-element GPU buffer that the per-step picker
writes (see :func:`solver_phoenx_kernels._pick_threads_per_world_kernel`).
This script sweeps representative scenes across the four meaningful
configurations:

  * ``"auto"``    -- the per-step adaptive picker decides;
  * ``32`` (forced) -- legacy one-warp-per-world layout (current default);
  * ``16`` (forced) -- two worlds per warp;
  * ``8``  (forced) -- four worlds per warp (gated tight in auto).

For each (scene, tpw) pair it captures a single ``simulate_one_frame``
call into a CUDA graph, replays it ``--n_runs`` times per trial across
``--trials`` trials, and reports the min and median wall-clock cost.
The min-of-trials is the cleanest signal; slow trials are usually
contention noise.

When to re-run
--------------
  * After tuning the picker thresholds in
    ``_pick_threads_per_world_kernel`` (verify the chosen tpw matches
    what the static-best column shows, on every scene).
  * After a kernel rewrite that touches the fast-tail path (verify
    the gain envelope hasn't shifted).
  * On a new GPU (``sm_count`` drives the saturation threshold).

How to interpret
----------------
The "auto vs tpw=32" column is the headline number. Positive ->
the picker found a better tpw than the legacy default. Watch for:

  * Auto matching forced-best within ~1% on every scene -- the picker
    isn't leaving wins on the table.
  * Auto never regressing more than ~2% vs forced=32 -- the picker
    isn't being too aggressive.

Reference numbers on RTX PRO 6000 (sm_count=188), with the heuristic
landed in the initial adaptive-tpw commit::

    h1_flat 64 worlds   -> auto pins tpw=32, neutral
    h1_flat 512 worlds  -> auto pins tpw=32, neutral
    h1_flat 2048 worlds -> auto picks tpw=16, ~+5%
    h1_flat 4096 worlds -> auto picks tpw=16, neutral / borderline
    h1_flat 8192 worlds -> auto picks tpw=16, ~+8-12%
    tower 32 worlds     -> auto pins tpw=32, neutral

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.bench_threads_per_world

Override the scene list or replay count via the ``--scenes`` /
``--n_runs`` flags. Nothing is persisted; results print to the
terminal.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.scenarios import h1_flat, tower
from newton._src.solvers.phoenx.solver import SolverPhoenX


def _extract_solver(handle) -> SolverPhoenX:
    """Reach into a :class:`SceneHandle` to grab its underlying solver.

    The scenario builders hide the solver inside the
    ``simulate_one_frame`` closure; we walk the closure cells to find
    it. Brittle if the closure shape changes, but the existing
    scenarios all follow the same pattern and changing it would break
    the benchmark dashboard too.
    """
    fn = handle.simulate_one_frame
    if fn.__closure__ is None:
        raise RuntimeError("simulate_one_frame has no closure")
    for cell in fn.__closure__:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, SolverPhoenX):
            return value
    raise RuntimeError("SolverPhoenX not found in simulate_one_frame closure")


def _force_tpw(solver: SolverPhoenX, tpw) -> None:
    """Apply ``tpw`` to a freshly-built solver after construction.

    The scenarios don't expose the ``threads_per_world`` constructor
    argument; we mutate the underlying state directly. ``"auto"``
    re-enables the per-step picker; an integer pins the value. The
    fast-tail kernels read ``_tpw_choice`` every launch, so this
    takes effect on the next ``solver.step()``.
    """
    if tpw == "auto":
        solver.world._tpw_auto = True
    else:
        solver.world._tpw_auto = False
        solver.world._tpw_choice.assign([int(tpw)])


def _build_h1(num_worlds: int, tpw, *, substeps: int, solver_iterations: int):
    """h1_flat fleet builder. Returns ``(solver, simulate_one_frame)``."""
    handle = h1_flat.build(
        num_worlds=num_worlds,
        solver_name="phoenx",
        substeps=substeps,
        solver_iterations=solver_iterations,
    )
    solver = _extract_solver(handle)
    _force_tpw(solver, tpw)
    return solver, handle.simulate_one_frame


def _build_tower(num_worlds: int, tpw, *, substeps: int, solver_iterations: int):
    """tower (multi_world layout) builder. Returns ``(solver, simulate_one_frame)``."""
    handle = tower.build(
        num_worlds=num_worlds,
        solver_name="phoenx",
        substeps=substeps,
        solver_iterations=solver_iterations,
        step_layout="multi_world",
    )
    solver = _extract_solver(handle)
    _force_tpw(solver, tpw)
    return solver, handle.simulate_one_frame


def _bench(simulate: Callable[[], None], n_runs: int, warmup: int, trials: int) -> tuple[float, float]:
    """Capture ``simulate`` into a CUDA graph; replay ``n_runs`` per trial.

    Returns ``(min_ms, median_ms)`` across ``trials`` trials. The min
    is the cleanest signal for a kernel-level micro-bench: contention
    inflates a trial but never deflates one.
    """
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


def _run_scene(
    label: str,
    builder: Callable,
    *,
    num_worlds: int,
    n_runs: int,
    warmup: int,
    trials: int,
    substeps: int,
    solver_iterations: int,
) -> None:
    print(f"\n=== {label} (num_worlds={num_worlds}, n_runs={n_runs}, trials={trials}) ===")
    results: dict[str, tuple[float, float, int]] = {}
    for tpw in ("auto", 32, 16, 8):
        solver, simulate = builder(
            num_worlds=num_worlds,
            tpw=tpw,
            substeps=substeps,
            solver_iterations=solver_iterations,
        )
        # A few real frames so contacts populate and the picker has a
        # current colour-stat snapshot to react to.
        for _ in range(5):
            simulate()
        wp.synchronize_device()
        chosen = int(solver.world._tpw_choice.numpy()[0]) if tpw == "auto" else int(tpw)
        min_ms, med_ms = _bench(simulate, n_runs=n_runs, warmup=warmup, trials=trials)
        results[str(tpw)] = (min_ms, med_ms, chosen)
        per_frame_us = 1000.0 * min_ms / n_runs
        print(
            f"  tpw={tpw!s:>5s}  chosen={chosen:>2d}  "
            f"min={min_ms:8.2f} ms  med={med_ms:8.2f} ms  "
            f"({per_frame_us:7.2f} us/frame @min)"
        )
    base_min = results["32"][0]
    print(f"  (rel. to forced tpw=32 baseline of {base_min:.2f} ms)")
    for key, (m, _, _) in results.items():
        if key == "32":
            continue
        speedup = base_min / m if m > 0 else float("nan")
        print(f"    tpw={key:>5s}: {speedup:.3f}x  ({100 * (speedup - 1):+.1f}%)")


_SCENE_REGISTRY: dict[str, tuple[Callable, str, int]] = {
    # name -> (builder, label, num_worlds)
    "h1_64": (_build_h1, "h1_flat", 64),
    "h1_512": (_build_h1, "h1_flat", 512),
    "h1_2048": (_build_h1, "h1_flat", 2048),
    "h1_4096": (_build_h1, "h1_flat", 4096),
    "h1_8192": (_build_h1, "h1_flat", 8192),
    "tower_32": (_build_tower, "tower", 32),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["h1_64", "h1_512", "h1_2048", "h1_4096", "h1_8192", "tower_32"],
        choices=sorted(_SCENE_REGISTRY.keys()),
        help="Scenes to bench (default: full sweep across h1_flat sizes + tower_32).",
    )
    parser.add_argument(
        "--n_runs", type=int, default=200, help="Graph replays per trial. Higher -> less noise, longer runtime."
    )
    parser.add_argument("--warmup", type=int, default=16, help="Frames stepped before capture so contacts settle.")
    parser.add_argument("--trials", type=int, default=3, help="Independent trials per (scene, tpw) -- min taken.")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--solver_iterations", type=int, default=8)
    args = parser.parse_args()

    wp.init()
    print(f"device: {wp.get_device()}  sm_count={getattr(wp.get_device(), 'sm_count', 'N/A')}")

    for scene_key in args.scenes:
        builder, label, num_worlds = _SCENE_REGISTRY[scene_key]
        # h1_flat at 8192 is heavy; halve runs unless the user overrode.
        n_runs = args.n_runs if num_worlds < 8192 else max(50, args.n_runs // 2)
        _run_scene(
            f"{label} {scene_key}",
            builder,
            num_worlds=num_worlds,
            n_runs=n_runs,
            warmup=args.warmup,
            trials=args.trials,
            substeps=args.substeps,
            solver_iterations=args.solver_iterations,
        )


if __name__ == "__main__":
    main()
