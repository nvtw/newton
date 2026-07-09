# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure maximal-coordinate PhoenX convergence on a heavy-on-light stack.

The instantaneous force from an iterative contact solve oscillates around the
supported load and can make a single final-frame check look much better or
worse than the actual solve. This benchmark therefore reports time-averaged
contact loads together with RMS body speed, minimum separation, and maximum
lateral drift over a fixed steady-state window.

Every physics frame is CUDA-graph captured. Host readback is intentionally
limited to the diagnostic quality window and is excluded from the throughput
measurement.

Examples::

    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_high_mass_ratio

    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_high_mass_ratio \
        --mass-ratios 100 400 --settings 5x4 8x4 8x8 --output /tmp/high_ratio.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import asdict, dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.tests.test_high_mass_ratio import GRAVITY, HE, _plane_pair_fz_to_body
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene


@dataclass(frozen=True)
class SolverSetting:
    """One temporal/GS work point on the convergence frontier."""

    substeps: int
    solver_iterations: int


@dataclass(frozen=True)
class BenchResult:
    """Steady-window physical metrics and graph-replay throughput."""

    mass_ratio: float
    substeps: int
    solver_iterations: int
    velocity_iterations: int
    sor_boost: float
    settle_frames: int
    sample_frames: int
    measure_frames: int
    fps: float
    finite: bool
    stack_intact: bool
    max_top_xy_m: float | None
    min_center_separation_m: float | None
    bottom_speed_rms_m_s: float | None
    top_speed_rms_m_s: float | None
    top_load_relative_error: float | None
    plane_load_relative_error: float | None


def _parse_setting(value: str) -> SolverSetting:
    try:
        substeps_text, iterations_text = value.lower().split("x", maxsplit=1)
        substeps = int(substeps_text)
        iterations = int(iterations_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("settings must have the form SUBSTEPSxITERATIONS, for example 8x4") from exc
    if substeps < 1 or iterations < 1:
        raise argparse.ArgumentTypeError("substeps and iterations must both be positive")
    return SolverSetting(substeps=substeps, solver_iterations=iterations)


def _build_scene(
    *,
    mass_ratio: float,
    setting: SolverSetting,
    velocity_iterations: int,
    sor_boost: float,
) -> tuple[_PhoenXScene, int, int]:
    scene = _PhoenXScene(
        substeps=setting.substeps,
        solver_iterations=setting.solver_iterations,
        velocity_iterations=velocity_iterations,
        step_layout="single_world",
        prepare_refresh_stride=1,
    )
    scene.add_ground_plane()
    bottom = scene.add_box(
        position=(0.0, 0.0, HE + 0.05),
        half_extents=(HE, HE, HE),
        mass=1.0,
    )
    top = scene.add_box(
        position=(0.0, 0.0, 3.0 * HE + 0.05),
        half_extents=(HE, HE, HE),
        mass=mass_ratio,
    )
    scene.finalize()
    scene.world.sor_boost = sor_boost
    return scene, bottom, top


def _run_one(
    *,
    mass_ratio: float,
    setting: SolverSetting,
    velocity_iterations: int,
    sor_boost: float,
    settle_frames: int,
    sample_frames: int,
    measure_frames: int,
) -> BenchResult:
    scene, bottom, top = _build_scene(
        mass_ratio=mass_ratio,
        setting=setting,
        velocity_iterations=velocity_iterations,
        sor_boost=sor_boost,
    )

    for _ in range(settle_frames):
        scene.step()

    samples = np.empty((sample_frames, 6), dtype=np.float64)
    for frame in range(sample_frames):
        scene.step()
        p_bottom = scene.body_position(bottom)
        p_top = scene.body_position(top)
        v_bottom = scene.body_velocity(bottom)
        v_top = scene.body_velocity(top)
        force_top, _, _ = scene.gather_contact_wrench_on_body(top)
        samples[frame] = (
            np.hypot(p_top[0], p_top[1]),
            p_top[2] - p_bottom[2],
            np.linalg.norm(v_bottom),
            np.linalg.norm(v_top),
            force_top[2],
            _plane_pair_fz_to_body(scene, bottom),
        )

    wp.synchronize_device(scene.device)
    start = time.perf_counter()
    for _ in range(measure_frames):
        scene.step()
    wp.synchronize_device(scene.device)
    elapsed = time.perf_counter() - start

    finite = bool(np.isfinite(samples).all())
    max_top_xy = float(np.max(samples[:, 0])) if finite else None
    min_separation = float(np.min(samples[:, 1])) if finite else None
    bottom_speed_rms = float(np.sqrt(np.mean(np.square(samples[:, 2])))) if finite else None
    top_speed_rms = float(np.sqrt(np.mean(np.square(samples[:, 3])))) if finite else None
    expected_top_load = mass_ratio * GRAVITY
    expected_plane_load = (1.0 + mass_ratio) * GRAVITY
    top_load_error = abs(float(np.mean(samples[:, 4])) - expected_top_load) / expected_top_load if finite else None
    plane_load_error = (
        abs(float(np.mean(samples[:, 5])) - expected_plane_load) / expected_plane_load if finite else None
    )
    return BenchResult(
        mass_ratio=mass_ratio,
        substeps=setting.substeps,
        solver_iterations=setting.solver_iterations,
        velocity_iterations=velocity_iterations,
        sor_boost=sor_boost,
        settle_frames=settle_frames,
        sample_frames=sample_frames,
        measure_frames=measure_frames,
        fps=float(measure_frames / elapsed),
        finite=finite,
        stack_intact=bool(
            finite
            and min_separation is not None
            and max_top_xy is not None
            and min_separation > 1.8 * HE
            and max_top_xy < 0.5 * HE
        ),
        max_top_xy_m=max_top_xy,
        min_center_separation_m=min_separation,
        bottom_speed_rms_m_s=bottom_speed_rms,
        top_speed_rms_m_s=top_speed_rms,
        top_load_relative_error=top_load_error,
        plane_load_relative_error=plane_load_error,
    )


def _print_result(result: BenchResult) -> None:
    if not result.finite:
        print(
            f"ratio={result.mass_ratio:g}:1  {result.substeps}x{result.solver_iterations}  "
            f"fps={result.fps:8.1f}  finite=False"
        )
        return
    assert result.top_load_relative_error is not None
    assert result.plane_load_relative_error is not None
    assert result.bottom_speed_rms_m_s is not None
    assert result.top_speed_rms_m_s is not None
    assert result.max_top_xy_m is not None
    print(
        f"ratio={result.mass_ratio:g}:1  {result.substeps}x{result.solver_iterations}  "
        f"fps={result.fps:8.1f}  intact={result.stack_intact}  "
        f"load_err(top/plane)={100.0 * result.top_load_relative_error:5.1f}%/"
        f"{100.0 * result.plane_load_relative_error:5.1f}%  "
        f"speed_rms(bottom/top)={result.bottom_speed_rms_m_s:.4f}/"
        f"{result.top_speed_rms_m_s:.4f} m/s  xy={result.max_top_xy_m:.4f} m"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass-ratios", nargs="+", type=float, default=(100.0,))
    parser.add_argument(
        "--settings",
        nargs="+",
        type=_parse_setting,
        default=tuple(_parse_setting(value) for value in ("5x2", "5x4", "8x4", "8x8")),
    )
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--sor-boost", type=float, default=1.0)
    parser.add_argument("--settle-frames", type=int, default=120)
    parser.add_argument("--sample-frames", type=int, default=120)
    parser.add_argument("--measure-frames", type=int, default=240)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if not wp.is_cuda_available():
        raise RuntimeError("the PhoenX convergence benchmark requires CUDA graph capture")
    if any(ratio <= 0.0 for ratio in args.mass_ratios):
        parser.error("mass ratios must be positive")
    if args.velocity_iterations < 0:
        parser.error("velocity iterations must be non-negative")
    if args.settle_frames < 1 or args.sample_frames < 1 or args.measure_frames < 1:
        parser.error("frame counts must be positive")

    results = [
        _run_one(
            mass_ratio=float(mass_ratio),
            setting=setting,
            velocity_iterations=int(args.velocity_iterations),
            sor_boost=float(args.sor_boost),
            settle_frames=int(args.settle_frames),
            sample_frames=int(args.sample_frames),
            measure_frames=int(args.measure_frames),
        )
        for mass_ratio in args.mass_ratios
        for setting in args.settings
    ]
    for result in results:
        _print_result(result)

    if args.output is not None:
        payload = {
            "schema": "phoenx_high_mass_ratio_v1",
            "device": wp.get_device("cuda:0").name,
            "results": [asdict(result) for result in results],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
