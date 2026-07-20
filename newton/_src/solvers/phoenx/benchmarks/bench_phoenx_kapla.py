# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end FPS + stability benchmark for the Kapla tower.

Drives :class:`example_kapla_tower.Example` through a fixed warmup +
steady-state window and reports:

* **FPS** — frames per second over the steady-state window.
* **drift** — mean L2 displacement of every brick between the first
  and last steady-state frame. Smaller is more stable.
* **speed residuals** — final linear/angular motion after settling.
* **max_z** — tower top height after the run; should stay near the
  initial value (~2.79 for the default 1x1 grid at scale 0.1) for a
  stable settle.

A baseline JSON is checked into ``benchmarks/results/`` so later
optimisation steps can ``--check`` against it and fail loudly on
regression. ``--write-baseline`` records a new baseline.

Examples::

    # Run the default sweep and print results.
    python -m newton._src.solvers.phoenx.benchmarks.bench_phoenx_kapla

    # Single config, more frames, for noise-tight numbers.
    python -m newton._src.solvers.phoenx.benchmarks.bench_phoenx_kapla \\
        --mass-splitting on --frames 240 --warmup 60

    # Sweep cached contact prepare every second substep.
    python -m newton._src.solvers.phoenx.benchmarks.bench_phoenx_kapla \\
        --mass-splitting off --prepare-refresh-stride 2

    # Refresh the recorded baseline (after a verified improvement).
    python -m newton._src.solvers.phoenx.benchmarks.bench_phoenx_kapla \\
        --write-baseline

    # CI-style: assert no regression vs baseline (FPS within ±5 %,
    # drift within +50 %).
    python -m newton._src.solvers.phoenx.benchmarks.bench_phoenx_kapla --check
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np
import warp as wp

# Defer Example import so ``--help`` works on systems without a CUDA
# device.

_BASELINE_PATH = pathlib.Path(__file__).parent / "results" / "phoenx_kapla_baseline.json"

# FPS tolerance for ``--check``: ±5 % matches typical run-to-run
# variance on a warm GPU.
_FPS_TOLERANCE: float = 0.05
# Drift tolerance: the noise on a settled tower is small (~mm); allow
# +50 % from the recorded baseline before failing.
_DRIFT_TOLERANCE: float = 0.50

# RTX PRO 6000 Blackwell ceilings measured by mini/bench_hardware_roofline.py.
_SEQUENTIAL_GBPS: float = 1489.14
_RANDOM_VEC4_GBPS: float = 1036.82
_FP32_TFLOPS: float = 87.810
_CONTACT_ITERATION_BYTES: float = 352.0
_CONTACT_ITERATION_FLOPS: float = 450.0


@dataclass
class BenchResult:
    """One measurement of one ``(mass_splitting, substeps, iters)`` config."""

    mass_splitting: bool
    substeps: int
    solver_iterations: int
    max_colored_partitions: int
    prepare_refresh_stride: int
    warmup_frames: int
    measured_frames: int
    fps: float
    mean_drift_m: float
    max_drift_m: float
    mean_speed_mps: float
    max_speed_mps: float
    max_angular_speed_radps: float
    max_z: float
    min_z: float
    finite: bool
    body_count: int
    contact_points: int
    contact_columns: int
    num_colors: int
    frame_ms: float
    logical_min_gbps: float
    sequential_bandwidth_percent: float
    random_vec4_bandwidth_percent: float
    estimated_tflops: float
    fp32_peak_percent: float
    roofline_basis: str
    blocks_per_sm: int
    colored_contact_layout: bool


class _HeadlessViewer:
    """Stub viewer that satisfies the :class:`Example` API without rendering."""

    def __init__(self):
        # Mimic the live viewer's camera.pos so the camera-collider
        # update has a deterministic position.
        self.camera = SimpleNamespace(pos=SimpleNamespace(x=1.2, y=0.0, z=0.4))

    def set_model(self, model):
        pass

    def set_camera(self, **kwargs):
        pos = kwargs.get("pos")
        if pos is not None:
            self.camera.pos.x = float(pos[0])
            self.camera.pos.y = float(pos[1])
            self.camera.pos.z = float(pos[2])

    def begin_frame(self, t):
        pass

    def log_state(self, state):
        pass

    def log_contacts(self, contacts, state):
        pass

    def end_frame(self):
        pass


class _HeadlessArgs:
    pass


def _run_one(
    *,
    mass_splitting: bool,
    substeps: int,
    solver_iterations: int,
    max_colored_partitions: int,
    prepare_refresh_stride: int,
    warmup_frames: int,
    measured_frames: int,
    grid_dims: tuple[int, int],
    blocks_per_sm: int,
    colored_contact_layout: bool,
) -> BenchResult:
    """Build the scene, run warmup + steady-state, return measurements."""
    # Import here so the module loads cheap (parser help, etc.).
    from newton._src.solvers.phoenx.examples import example_kapla_tower as ek

    ek.TOWER_GRID_DIMS = grid_dims
    ek.ENABLE_MASS_SPLITTING = mass_splitting
    ek.MASS_SPLITTING_MAX_COLORED_PARTITIONS = max_colored_partitions
    ek.USE_COLORED_CONTACT_HEADERS = colored_contact_layout
    ek.USE_COLORED_CONTACT_ROWS = colored_contact_layout

    if mass_splitting and prepare_refresh_stride != 1:
        raise ValueError("prepare_refresh_stride > 1 is only supported with --mass-splitting off")

    ex = ek.Example(_HeadlessViewer(), _HeadlessArgs())
    ex.world._singleworld_total_threads = blocks_per_sm * ex.device.sm_count * 32
    ex.world.solver_iterations = solver_iterations
    ex.world.substeps = substeps
    ex.world.prepare_refresh_stride = prepare_refresh_stride
    # Re-capture the per-frame graph so the solver-config change takes
    # effect.
    ex.graph = None
    with wp.ScopedCapture(device=ex.device) as capture:
        ex.simulate()
    ex.graph = capture.graph

    for _ in range(warmup_frames):
        ex.step()
    wp.synchronize_device(ex.device)

    pos_start = ex.bodies.position.numpy().copy()
    t0 = time.perf_counter()
    for _ in range(measured_frames):
        ex.step()
    wp.synchronize_device(ex.device)
    elapsed = time.perf_counter() - t0
    pos_end = ex.bodies.position.numpy()
    velocity_end = ex.bodies.velocity.numpy()
    angular_velocity_end = ex.bodies.angular_velocity.numpy()
    motion_type = ex.bodies.motion_type.numpy()

    fps = measured_frames / elapsed
    dynamic = motion_type >= 2
    drift = np.linalg.norm(pos_end[dynamic] - pos_start[dynamic], axis=1)
    speeds = np.linalg.norm(velocity_end[dynamic], axis=1)
    angular_speeds = np.linalg.norm(angular_velocity_end[dynamic], axis=1)
    zs = pos_end[dynamic, 2]
    report = ex.world.step_report()
    contact_points = int(ex.contacts.rigid_contact_count.numpy()[0])
    iteration_rate = contact_points * solver_iterations * substeps * ex.steps_per_frame * fps
    logical_min_gbps = iteration_rate * _CONTACT_ITERATION_BYTES / 1.0e9
    estimated_tflops = iteration_rate * _CONTACT_ITERATION_FLOPS / 1.0e12
    return BenchResult(
        mass_splitting=mass_splitting,
        substeps=substeps,
        solver_iterations=solver_iterations,
        max_colored_partitions=max_colored_partitions,
        prepare_refresh_stride=prepare_refresh_stride,
        warmup_frames=warmup_frames,
        measured_frames=measured_frames,
        fps=float(fps),
        mean_drift_m=float(drift.mean()),
        max_drift_m=float(drift.max()),
        mean_speed_mps=float(speeds.mean()),
        max_speed_mps=float(speeds.max()),
        max_angular_speed_radps=float(angular_speeds.max()),
        max_z=float(zs.max()),
        min_z=float(zs.min()),
        finite=bool(np.isfinite(pos_end).all()),
        body_count=int(ex.model.body_count),
        contact_points=contact_points,
        contact_columns=report.num_contact_columns,
        num_colors=report.num_colors,
        frame_ms=1000.0 / fps,
        logical_min_gbps=logical_min_gbps,
        sequential_bandwidth_percent=100.0 * logical_min_gbps / _SEQUENTIAL_GBPS,
        random_vec4_bandwidth_percent=100.0 * logical_min_gbps / _RANDOM_VEC4_GBPS,
        estimated_tflops=estimated_tflops,
        fp32_peak_percent=100.0 * estimated_tflops / _FP32_TFLOPS,
        roofline_basis="352 B and 450 FLOP per final contact-point iteration; useful-work estimate, no GPU counters",
        blocks_per_sm=blocks_per_sm,
        colored_contact_layout=colored_contact_layout,
    )


def _format_row(r: BenchResult) -> str:
    label = (
        f"ms={r.mass_splitting!s:<5} sub={r.substeps} it={r.solver_iterations} "
        f"cap={r.max_colored_partitions} prep={r.prepare_refresh_stride}"
    )
    return (
        f"  {label:<24} fps={r.fps:6.2f}  drift mean={r.mean_drift_m:.4f}m "
        f"max={r.max_drift_m:.4f}m  speed={r.mean_speed_mps:.4f}/{r.max_speed_mps:.4f}m/s "
        f"max_w={r.max_angular_speed_radps:.3f}rad/s  z=[{r.min_z:.3f}, {r.max_z:.3f}]  "
        f"contacts={r.contact_points}/{r.contact_columns} colors={r.num_colors} "
        f"BW={r.logical_min_gbps:.1f}GB/s ({r.sequential_bandwidth_percent:.1f}% seq) "
        f"finite={r.finite}"
    )


def _default_sweep() -> list[dict]:
    """Configs covered by the default sweep.

    Each entry mirrors the production knobs Kapla cares about: mass
    splitting toggle (both states), and the
    ``(substeps, solver_iterations)`` pair the example ships with.
    """
    return [
        {"mass_splitting": False, "substeps": 4, "solver_iterations": 5},
        {"mass_splitting": False, "substeps": 4, "solver_iterations": 10},
        {"mass_splitting": True, "substeps": 4, "solver_iterations": 5},
        {"mass_splitting": True, "substeps": 4, "solver_iterations": 10},
    ]


def _baseline_key(r: BenchResult) -> tuple[str, ...]:
    """Baseline key. Preserve legacy stride-1 keys for checked-in data."""
    key = (str(r.mass_splitting), str(r.substeps), str(r.solver_iterations))
    if r.max_colored_partitions != 8:
        key = (*key, f"cap={r.max_colored_partitions}")
    if r.prepare_refresh_stride != 1:
        key = (*key, f"prep={r.prepare_refresh_stride}")
    return key


def _check_against_baseline(results: list[BenchResult]) -> int:
    """Return 0 if every config is within tolerance of the recorded
    baseline; non-zero on any regression."""
    if not _BASELINE_PATH.exists():
        print(f"[bench_phoenx_kapla] no baseline at {_BASELINE_PATH}; nothing to check", file=sys.stderr)
        return 2
    with _BASELINE_PATH.open() as f:
        baseline = {tuple(map(str, r["key"])): r["result"] for r in json.load(f)}
    rc = 0
    for r in results:
        key = _baseline_key(r)
        if key not in baseline:
            print(f"[bench_phoenx_kapla] {key} not in baseline", file=sys.stderr)
            continue
        b = baseline[key]
        fps_delta = (r.fps - b["fps"]) / b["fps"]
        drift_delta = (r.mean_drift_m - b["mean_drift_m"]) / max(b["mean_drift_m"], 1e-6)
        ok_fps = fps_delta > -_FPS_TOLERANCE
        ok_drift = drift_delta < _DRIFT_TOLERANCE
        verdict = "OK" if (ok_fps and ok_drift) else "REGRESS"
        print(
            f"  [{verdict}] ms={r.mass_splitting!s:<5} sub={r.substeps} it={r.solver_iterations} "
            f"prep={r.prepare_refresh_stride} "
            f"fps Δ {fps_delta:+.1%} (now {r.fps:.2f}, base {b['fps']:.2f})  "
            f"drift Δ {drift_delta:+.1%} (now {r.mean_drift_m:.4f}, base {b['mean_drift_m']:.4f})"
        )
        if not (ok_fps and ok_drift):
            rc = 1
    return rc


def _write_baseline(results: list[BenchResult]) -> None:
    payload = [
        {
            "key": list(_baseline_key(r)),
            "result": asdict(r),
        }
        for r in results
    ]
    _BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _BASELINE_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[bench_phoenx_kapla] baseline written to {_BASELINE_PATH}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mass-splitting",
        choices=["on", "off", "both"],
        default="both",
        help="Run one or both mass-splitting configurations (default both).",
    )
    p.add_argument(
        "--substeps",
        type=int,
        default=None,
        help="Override solver substeps. Default uses the sweep.",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Override solver_iterations. Default uses the sweep.",
    )
    p.add_argument(
        "--max-colored-partitions",
        type=int,
        default=8,
        help="True GS colors retained before mass-splitting overflow (default 8).",
    )
    p.add_argument("--warmup", type=int, default=60, help="Warmup frames (default 60).")
    p.add_argument("--frames", type=int, default=120, help="Measured frames (default 120).")
    p.add_argument(
        "--prepare-refresh-stride",
        type=int,
        default=1,
        help=(
            "Refresh cached rigid-contact prepare data every N substeps (default 1). "
            "Only 2 is supported for contact-only non-mass-splitting runs."
        ),
    )
    p.add_argument("--grid", type=int, default=1, help="Tower grid edge length (default 1 = 1x1).")
    p.add_argument("--blocks-per-sm", type=int, default=8, help="Persistent-grid blocks per GPU SM.")
    p.add_argument(
        "--colored-contact-layout",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use color-ordered contact headers and rows; auto enables them with mass splitting.",
    )
    p.add_argument("--write-baseline", action="store_true", help="Persist results as the new baseline.")
    p.add_argument("--check", action="store_true", help="Compare to baseline; exit nonzero on regression.")
    p.add_argument("--json", action="store_true", help="Print results as JSON instead of a table.")
    args = p.parse_args(argv)

    if args.blocks_per_sm < 1:
        p.error("--blocks-per-sm must be >= 1")
    if args.max_colored_partitions < 0:
        p.error("--max-colored-partitions must be >= 0")

    if args.prepare_refresh_stride != 1 and args.mass_splitting != "off":
        p.error("--prepare-refresh-stride > 1 requires --mass-splitting off")

    if args.mass_splitting == "both" and args.substeps is None and args.iters is None:
        configs = _default_sweep()
    else:
        ms_options = (
            [True] if args.mass_splitting == "on" else [False] if args.mass_splitting == "off" else [False, True]
        )
        substeps = args.substeps if args.substeps is not None else 4
        iters = args.iters if args.iters is not None else 10
        configs = [{"mass_splitting": ms, "substeps": substeps, "solver_iterations": iters} for ms in ms_options]

    results: list[BenchResult] = []
    grid_dims = (args.grid, args.grid)
    for cfg in configs:
        colored_contact_layout = (
            cfg["mass_splitting"] if args.colored_contact_layout == "auto" else args.colored_contact_layout == "on"
        )
        if colored_contact_layout and not cfg["mass_splitting"]:
            p.error("--colored-contact-layout on requires --mass-splitting on")
        r = _run_one(
            mass_splitting=cfg["mass_splitting"],
            substeps=cfg["substeps"],
            solver_iterations=cfg["solver_iterations"],
            max_colored_partitions=args.max_colored_partitions,
            prepare_refresh_stride=args.prepare_refresh_stride,
            warmup_frames=args.warmup,
            measured_frames=args.frames,
            grid_dims=grid_dims,
            blocks_per_sm=args.blocks_per_sm,
            colored_contact_layout=colored_contact_layout,
        )
        results.append(r)
        if not args.json:
            print(_format_row(r))

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))

    if args.write_baseline:
        _write_baseline(results)

    if args.check:
        return _check_against_baseline(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
