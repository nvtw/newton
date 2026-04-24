# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hand-triggered PhoenX vs. MuJoCo-Warp benchmark sweep.

Single-GPU, local. Launch with::

    python -m newton._src.solvers.phoenx.benchmarks.run_benchmarks

By default runs a small sweep (3 world counts, 1 substep count)
so a first run finishes in a few minutes. Pass ``--full-sweep`` to
expand to Dylan Turpin's nightly grid.

Writes one ``points.jsonl`` row per (scenario, solver, num_worlds,
substeps, iterations) config to
``newton/_src/solvers/phoenx/benchmarks/results/`` (next to the
Chart.js dashboard). No network, no git ops -- you decide whether
to ship the artefacts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import warp as wp

from newton._src.solvers.phoenx.benchmarks.runner import (
    reset_gpu_between_runs,
    run_one,
)
from newton._src.solvers.phoenx.benchmarks.scenarios import SCENARIOS

RESULTS_DIR = Path(__file__).parent / "results"
POINTS_PATH = RESULTS_DIR / "points.jsonl"
RUNS_PATH = RESULTS_DIR / "runs.jsonl"


# ---------------------------------------------------------------------------
# GPU lock banner -- intentionally obnoxious
# ---------------------------------------------------------------------------
#
# Clock locking on Linux requires ``sudo nvidia-smi --lock-gpu-clocks=...``
# so we can't do it for the user. But timings without it vary 2-5x
# run-to-run on stock boost-clock GPUs, which makes regression
# detection impossible. Print a banner before any work so the user
# can ctrl-C out, lock clocks, and rerun.


_BANNER = """
\033[1;33m===============================================================
WARNING: GPU CLOCK LOCKING STRONGLY RECOMMENDED
===============================================================
Stable timings require pinned GPU clocks. Check your base clock:

    nvidia-smi --query-gpu=clocks.base.graphics --format=csv,noheader,nounits

Then lock (requires sudo):

    sudo nvidia-smi -pm 1
    sudo nvidia-smi --lock-gpu-clocks=<base>

Reset when done:

    sudo nvidia-smi --reset-gpu-clocks

Without this, expect 2-5x run-to-run variance. Results remain
comparable within a single run but NOT across runs.
===============================================================
\033[0m
"""


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _gpu_name() -> str:
    device = wp.get_device()
    # ``wp.context.Device`` exposes ``arch`` / ``name`` on recent
    # Warp; fall back to str() on older ones.
    return getattr(device, "name", None) or str(device)


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Sweep plan
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """Cartesian product over scenes, solvers, num_worlds, substeps."""

    scenarios: list[str] = field(default_factory=lambda: ["g1_flat", "h1_flat"])
    solvers: list[str] = field(default_factory=lambda: ["phoenx", "mujoco"])
    num_worlds: list[int] = field(default_factory=lambda: [1024, 4096, 16384])
    substeps: list[int] = field(default_factory=lambda: [4])
    solver_iterations: list[int] = field(default_factory=lambda: [8])
    warmup_frames: int = 16
    measure_frames: int = 64

    def iter_configs(self):
        for scene in self.scenarios:
            for solver in self.solvers:
                for nw in self.num_worlds:
                    for ss in self.substeps:
                        for it in self.solver_iterations:
                            yield (scene, solver, nw, ss, it)


_FULL_SWEEP = SweepConfig(
    scenarios=["g1_flat", "h1_flat"],
    solvers=["phoenx", "mujoco"],
    # Dylan's nightly spans 1024-131072; capped for reasonable local runtime.
    num_worlds=[1024, 2048, 4096, 8192, 16384, 32768],
    substeps=[2, 4],
    solver_iterations=[8],
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _banner_and_confirm(skip_prompt: bool) -> None:
    print(_BANNER, file=sys.stderr)
    if skip_prompt:
        return
    try:
        input("Press Enter to continue, or Ctrl-C to lock clocks first: ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _common_metadata() -> dict:
    return {
        "timestamp": _iso_now(),
        "commit_short": _git_commit_short(),
        "gpu": _gpu_name(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def run_sweep(cfg: SweepConfig, clear_existing: bool) -> None:
    """Run every (scene, solver, num_worlds, substeps, iter) combo in
    ``cfg`` and append a ``points.jsonl`` row per config.

    Failures (OOM, missing asset) are caught per config so the sweep
    keeps going; the failing row still appears in JSONL with
    ``ok=False`` and the error message.
    """
    run_id = _iso_now().replace(":", "").replace("-", "")
    meta = _common_metadata()
    meta["run_id"] = run_id

    if clear_existing and POINTS_PATH.exists():
        POINTS_PATH.unlink()

    configs = list(cfg.iter_configs())
    print(f"[bench] plan: {len(configs)} configs, GPU={meta['gpu']}")
    print(f"[bench] writing to {POINTS_PATH}")
    t_sweep = 0.0
    for i, (scene, solver, nw, ss, it) in enumerate(configs, 1):
        label = f"{scene} / {solver} / num_worlds={nw} / substeps={ss} / iter={it}"
        print(f"[bench] ({i}/{len(configs)}) {label}", flush=True)
        row = {**meta}
        row["scenario"] = scene
        row["solver"] = solver
        row["num_worlds"] = nw
        row["substeps"] = ss
        row["solver_iterations"] = it
        try:
            reset_gpu_between_runs()
            handle = SCENARIOS[scene].build(
                num_worlds=nw,
                solver_name=solver,
                substeps=ss,
                solver_iterations=it,
            )
            metrics = run_one(
                handle,
                warmup_frames=cfg.warmup_frames,
                measure_frames=cfg.measure_frames,
            )
            # ``metrics`` already has scenario / solver / substeps /
            # etc.; ``row`` already has them too, so ``metrics``
            # wins on collisions.
            row.update(metrics)
            t_sweep += metrics["elapsed_s"]
            print(
                f"  -> env_fps={metrics['env_fps']:,.0f}"
                f"  ms/step={metrics['ms_per_step']:.3f}"
                f"  gpu_used_gb={metrics['gpu_used_gb']:.2f}",
                flush=True,
            )
            del handle
        except Exception as exc:
            # OOM, missing assets, solver mismatch -- keep the sweep
            # alive and record what we know.
            err = f"{type(exc).__name__}: {exc}"
            row["ok"] = False
            row["error"] = err
            row.setdefault("elapsed_s", 0.0)
            row.setdefault("env_fps", 0.0)
            row.setdefault("ms_per_step", 0.0)
            row.setdefault("gpu_used_gb", 0.0)
            row.setdefault("gpu_total_gb", 0.0)
            print(f"  -> FAIL: {err}", flush=True)
            traceback.print_exc(file=sys.stderr)
        _append_jsonl(POINTS_PATH, row)

    # runs.jsonl: one summary row per sweep invocation.
    run_row = {
        **meta,
        "run_id": run_id,
        "config_count": len(configs),
        "total_measure_s": t_sweep,
        "warmup_frames": cfg.warmup_frames,
        "measure_frames": cfg.measure_frames,
    }
    _append_jsonl(RUNS_PATH, run_row)

    print(f"\n[bench] done. total measurement time: {t_sweep:.1f}s")
    print(f"[bench] results: {POINTS_PATH}")
    print("[bench] view dashboard:")
    print(f"  cd {POINTS_PATH.parent.parent}")
    print("  python -m http.server 8000")
    print("  then open http://localhost:8000/dashboard/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hand-triggered PhoenX vs MuJoCo-Warp benchmark sweep."
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Run Dylan Turpin's full nightly grid (capped to 32768 worlds locally).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help=f"Subset of scenarios to run (default: both). Available: {', '.join(SCENARIOS)}",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=None,
        choices=["phoenx", "mujoco"],
        help="Subset of solvers to run (default: both).",
    )
    parser.add_argument(
        "--num-worlds",
        type=int,
        nargs="+",
        default=None,
        help="Override num_worlds sweep.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        nargs="+",
        default=None,
        help="Override substep sweep.",
    )
    parser.add_argument(
        "--solver-iterations",
        type=int,
        nargs="+",
        default=None,
        help="Override PGS iteration sweep.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=16,
        help="Warmup frames per config (discarded).",
    )
    parser.add_argument(
        "--measure-frames",
        type=int,
        default=64,
        help="Measured frames per config.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing points.jsonl instead of truncating it.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the GPU-lock confirmation prompt.",
    )
    args = parser.parse_args(argv)

    cfg = _FULL_SWEEP if args.full_sweep else SweepConfig()
    if args.scenarios is not None:
        cfg.scenarios = list(args.scenarios)
    if args.solvers is not None:
        cfg.solvers = list(args.solvers)
    if args.num_worlds is not None:
        cfg.num_worlds = list(args.num_worlds)
    if args.substeps is not None:
        cfg.substeps = list(args.substeps)
    if args.solver_iterations is not None:
        cfg.solver_iterations = list(args.solver_iterations)
    cfg.warmup_frames = args.warmup_frames
    cfg.measure_frames = args.measure_frames

    _banner_and_confirm(skip_prompt=args.yes)

    wp.init()
    device = wp.get_device()
    if not device.is_cuda:
        print(
            "[bench] WARNING: running on CPU. Timings will be noisy and",
            "unrepresentative of production usage.",
            file=sys.stderr,
        )

    run_sweep(cfg, clear_existing=not args.append)
    return 0


if __name__ == "__main__":
    sys.exit(main())
