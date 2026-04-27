# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-scope memory + per-kernel timing profile for a PhoenX (or
MuJoCo) scene.

Answers two questions the aggregate bench can't:

  1. Where is the GPU memory going? Broken down into *model build*,
     *pipeline allocation*, *solver setup*, and *per-step scratch*
     (high-water mark of one graph-captured step after warmup).
  2. Where is each per-step millisecond going? Per-kernel breakdown
     from an *eager* (non-graph-captured) step under
     :class:`wp.ScopedTimer(cuda_filter=wp.TIMING_KERNEL)`.

Graph-captured replays are much faster but CUDA hides their
per-kernel timing (the Warp event API only sees the graph as a
single opaque entry). So for regression hunting we intentionally
run one eager step; the absolute ms numbers are ~5-10% higher than
the graph-captured steady state but the relative ranking is
reliable.

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.profile_phoenx \\
        --scenario h1_flat --solver phoenx --num-worlds 256

Writes a terminal report; nothing is persisted.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import sys
from collections import defaultdict
from dataclasses import dataclass

import warp as wp

# ---------------------------------------------------------------------------
# Memory scope helper
# ---------------------------------------------------------------------------


@dataclass
class MemoryScope:
    """Accumulates (used_current, used_high) deltas over a named scope.

    Usage::

        with MemoryScope("setup") as m:
            build_something()
        print(m.delta_gb, m.peak_gb)
    """

    label: str
    start_current: int = 0
    start_high: int = 0
    end_current: int = 0
    end_high: int = 0

    def __enter__(self) -> MemoryScope:
        wp.synchronize_device()
        device = wp.get_device()
        if device.is_cuda:
            self.start_current = wp.get_mempool_used_mem_current(device)
            self.start_high = wp.get_mempool_used_mem_high(device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        wp.synchronize_device()
        device = wp.get_device()
        if device.is_cuda:
            self.end_current = wp.get_mempool_used_mem_current(device)
            self.end_high = wp.get_mempool_used_mem_high(device)

    @property
    def delta_gb(self) -> float:
        return (self.end_current - self.start_current) / (1024**3)

    @property
    def peak_delta_gb(self) -> float:
        return (self.end_high - self.start_high) / (1024**3)


# ---------------------------------------------------------------------------
# Scenario drivers
# ---------------------------------------------------------------------------


@dataclass
class ProfileScene:
    """Holds everything needed to step a scene once for profiling."""

    name: str
    solver_name: str
    num_worlds: int
    # ``step()`` advances one frame. Simple callable so the profile
    # driver doesn't need to know about model / state swapping.
    step: callable


def _build_via_benchmarks_factory(
    scenario: str, num_worlds: int, solver: str, substeps: int, iterations: int
) -> ProfileScene:
    """Reuse the existing benchmarks/scenarios/*.py modules. They
    already produce a ``simulate_one_frame`` closure over their
    scene state -- perfect fit.
    """
    module = importlib.import_module(f"newton._src.solvers.phoenx.benchmarks.scenarios.{scenario}")
    handle = module.build(
        num_worlds=num_worlds,
        solver_name=solver,
        substeps=substeps,
        solver_iterations=iterations,
    )
    return ProfileScene(
        name=scenario,
        solver_name=solver,
        num_worlds=num_worlds,
        step=handle.simulate_one_frame,
    )


# ---------------------------------------------------------------------------
# Per-kernel timing aggregator
# ---------------------------------------------------------------------------


def _run_eager_kernel_timing(scene: ProfileScene, warmup: int = 4) -> dict[str, dict]:
    """Warm up, then run one eager step under
    :class:`wp.ScopedTimer(cuda_filter=wp.TIMING_KERNEL)` so every
    kernel launch inside ``scene.step`` is attributed.

    Aggregates by kernel name: ``{name: {calls, total_ms, max_ms,
    min_ms}}``. Returns the dict sorted by descending ``total_ms``.
    """
    for _ in range(warmup):
        scene.step()
    wp.synchronize_device()

    with wp.ScopedTimer(
        "step",
        synchronize=True,
        cuda_filter=wp.TIMING_KERNEL,
        print=False,
    ) as t:
        scene.step()

    agg: dict[str, dict] = defaultdict(lambda: {"calls": 0, "total_ms": 0.0, "max_ms": 0.0, "min_ms": 1e18})
    for r in t.timing_results:
        # Warp labels: "forward kernel <kernel_name_hash>". Strip the
        # common prefix so the report stays readable.
        name = r.name
        for prefix in ("forward kernel ", "backward kernel "):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        rec = agg[name]
        rec["calls"] += 1
        rec["total_ms"] += float(r.elapsed)
        rec["max_ms"] = max(rec["max_ms"], float(r.elapsed))
        rec["min_ms"] = min(rec["min_ms"], float(r.elapsed))
    # Descending total.
    return dict(sorted(agg.items(), key=lambda item: -item[1]["total_ms"]))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_header(scene: ProfileScene) -> None:
    device = wp.get_device()
    gpu_name = getattr(device, "name", None) or str(device)
    print("=" * 80)
    print(f"  scenario={scene.name}  solver={scene.solver_name}  num_worlds={scene.num_worlds}")
    print(f"  gpu={gpu_name}")
    print("=" * 80)


def _print_memory(scopes: list[MemoryScope]) -> None:
    print()
    print("-- GPU mempool (current delta / peak delta) --")
    total_delta = 0.0
    for s in scopes:
        print(f"  {s.label:<24} delta={s.delta_gb:+7.3f} GB   peak_delta={s.peak_delta_gb:+7.3f} GB")
        total_delta += s.delta_gb
    print(f"  {'TOTAL (sum of deltas)':<24}           {total_delta:+7.3f} GB")


def _print_kernels(agg: dict[str, dict], top_n: int, *, scene_name: str = "") -> None:
    print()
    print(f"-- per-kernel breakdown (eager, TIMING_KERNEL), top {top_n} --")
    total = sum(v["total_ms"] for v in agg.values())
    total_calls = sum(v["calls"] for v in agg.values())
    print(f"  total_kernel_ms={total:.3f}   total_calls={total_calls}   unique_kernels={len(agg)}")
    print(f"  {'rank':>4}  {'calls':>5}  {'total(ms)':>10}  {'avg(ms)':>9}  {'%':>6}  name")
    for i, (name, v) in enumerate(list(agg.items())[:top_n], 1):
        avg = v["total_ms"] / v["calls"]
        pct = v["total_ms"] / total * 100 if total > 0 else 0.0
        short_name = name[-70:] if len(name) > 70 else name
        print(f"  {i:>4}  {v['calls']:>5}  {v['total_ms']:>10.4f}  {avg:>9.4f}  {pct:>5.1f}%  {short_name}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def profile(
    scenario: str,
    solver: str,
    num_worlds: int,
    *,
    substeps: int = 4,
    iterations: int = 8,
    top_n: int = 20,
) -> None:
    device = wp.get_device()
    if not device.is_cuda:
        print("[profile] CUDA device required", file=sys.stderr)
        sys.exit(1)

    # Order matters: take scopes around the phases we want
    # attributed. Python-gc before each scope so stale references
    # from the previous run don't skew the delta.
    gc.collect()
    wp.synchronize_device()

    with MemoryScope("1. scene_build") as m_build:
        scene = _build_via_benchmarks_factory(scenario, num_worlds, solver, substeps, iterations)

    # Warmup pass (module JIT, lazy scratch allocations).
    with MemoryScope("2. warmup_2_frames") as m_warmup:
        for _ in range(2):
            scene.step()
        wp.synchronize_device()

    # Steady-state scratch during one additional measured eager step.
    with MemoryScope("3. one_eager_step") as m_step:
        scene.step()

    _print_header(scene)
    _print_memory([m_build, m_warmup, m_step])

    # Kernel breakdown on a subsequent eager step after another
    # warmup to reach steady-state.
    agg = _run_eager_kernel_timing(scene, warmup=4)
    _print_kernels(agg, top_n=top_n)

    # Free everything explicitly so subsequent profile() calls in
    # the same process don't double-count.
    del scene
    gc.collect()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile a phoenx / mujoco scene: memory + per-kernel ms.")
    parser.add_argument("--scenario", default="h1_flat")
    parser.add_argument("--solver", choices=["phoenx", "mujoco"], default="phoenx")
    parser.add_argument("--num-worlds", type=int, default=64)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--top", type=int, default=20, help="rows in the kernel table")
    args = parser.parse_args(argv)

    wp.init()
    profile(
        scenario=args.scenario,
        solver=args.solver,
        num_worlds=args.num_worlds,
        substeps=args.substeps,
        iterations=args.iterations,
        top_n=args.top,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
