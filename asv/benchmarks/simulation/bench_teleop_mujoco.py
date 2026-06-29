# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import newton.examples
from newton.examples.robot.example_robot_teleop_mujoco import Example as TeleopExample
from newton.viewer import ViewerNull


@dataclass(frozen=True)
class _TeleopMode:
    device: str
    mujoco_backend: str
    requires_cuda: bool = False
    requires_cuda_graph: bool = False


_TELEOP_MODES = {
    "mjwarp_cuda_graph": _TeleopMode(
        device="cuda:0",
        mujoco_backend="warp",
        requires_cuda=True,
        requires_cuda_graph=True,
    ),
    "mjwarp_cpu_eager": _TeleopMode(device="cpu", mujoco_backend="warp"),
    "mujoco_cpu_eager": _TeleopMode(device="cpu", mujoco_backend="cpu"),
}


def _skip_unavailable_mode(mode: _TeleopMode) -> None:
    if mode.requires_cuda and wp.get_cuda_device_count() == 0:
        raise SkipNotImplemented

    if mode.requires_cuda_graph:
        with wp.ScopedDevice(mode.device):
            if not wp.is_mempool_enabled(wp.get_device()):
                raise SkipNotImplemented


def _make_args(mode: _TeleopMode, num_frames: int):
    args = newton.examples.default_args(TeleopExample.create_parser())
    args.viewer = "null"
    args.headless = True
    args.test = True
    args.input = "scripted"
    args.num_frames = num_frames
    args.benchmark = False
    args.print_metrics = False
    args.mujoco_backend = mode.mujoco_backend
    # Latency runs must not include diagnostic device-to-host readbacks.
    args.metrics_interval = 0
    args.metrics_warmup_frames = 0
    args.sync_latency = True
    args.collect_stage_metrics = False
    args.render_shadows = False
    return args


class _TeleopMuJoCoBenchmark:
    """Benchmark the FR3 scripted control loop with MuJoCo solver backends.

    The scene reuses ``example_robot_teleop_mujoco``: a Franka FR3/Panda
    tracks a moving six-DoF end-effector target over a table-top manipulation
    scene while periodically actuating its gripper. Each measured frame covers
    scripted command generation, IK, joint-target writes, and a completed
    physics step. Rendering and physical input-device latency are intentionally
    excluded.

    GPU runs synchronize once at the control-loop boundary. This reports the
    latency visible to a synchronous 60 Hz controller without inserting
    artificial barriers between the stages being measured. Tracking readbacks
    are enabled only by the quality metrics so they do not perturb latency.
    """

    params = (tuple(_TELEOP_MODES.keys()),)
    param_names = ["mode"]
    repeat = 3
    number = 1
    rounds = 2
    timeout = 600

    num_frames = 300
    warmup_frames = 60

    def setup(self, mode: str) -> None:
        self.mode = _TELEOP_MODES[mode]
        _skip_unavailable_mode(self.mode)

        with wp.ScopedDevice(self.mode.device):
            args = _make_args(self.mode, self.num_frames)
            self.example = TeleopExample(ViewerNull(num_frames=self.num_frames), args)
            self._step_frames(self.warmup_frames)
            wp.synchronize_device(self.example.device)
            self._clear_measurements()

    def time_teleop_loop(self, mode: str) -> None:
        with wp.ScopedDevice(self.mode.device):
            self._step_frames(self.num_frames)

    def track_mean_loop_ms(self, mode: str) -> float:
        elapsed = self._measure_frames()
        return elapsed * 1000.0 / self.num_frames

    track_mean_loop_ms.unit = "ms/frame"

    def track_p95_loop_ms(self, mode: str) -> float:
        self._measure_frames()
        return self._summary("local_loop_ms")[1]

    track_p95_loop_ms.unit = "ms/frame"

    def track_frame_overrun_pct(self, mode: str) -> float:
        self._measure_frames()
        values = self.example.stats.values["local_loop_ms"]
        budget_ms = self.example.frame_dt * 1000.0
        overruns = sum(1 for value in values if value > budget_ms)
        return 100.0 * overruns / len(values)

    track_frame_overrun_pct.unit = "%"

    def track_mean_target_error_m(self, mode: str) -> float:
        self._measure_frames(collect_tracking=True)
        return self._summary("target_error_m")[0]

    track_mean_target_error_m.unit = "m"

    def track_mean_target_rotation_error_rad(self, mode: str) -> float:
        self._measure_frames(collect_tracking=True)
        return self._summary("target_rotation_error_rad")[0]

    track_mean_target_rotation_error_rad.unit = "rad"

    def teardown(self, mode: str) -> None:
        example = getattr(self, "example", None)
        if example is not None:
            example.close()
            del self.example

    def _step_frames(self, frame_count: int) -> None:
        for _ in range(frame_count):
            self.example.step()

    def _measure_frames(self, *, collect_tracking: bool = False) -> float:
        with wp.ScopedDevice(self.mode.device):
            self._clear_measurements()
            previous_interval = self.example.metrics_interval
            self.example.metrics_interval = 1 if collect_tracking else 0
            start = time.perf_counter()
            try:
                self._step_frames(self.num_frames)
                return time.perf_counter() - start
            finally:
                self.example.metrics_interval = previous_interval

    def _clear_measurements(self) -> None:
        self.example.clear_metrics()

    def _summary(self, name: str) -> tuple[float, float, float]:
        summary = self.example.stats.summary(name)
        if summary is None:
            raise RuntimeError(f"No teleop samples collected for {name!r}")
        return summary


class FastTeleopMuJoCo(_TeleopMuJoCoBenchmark):
    """Pull-request smoke benchmark for the CUDA graph teleop path."""

    params = (("mjwarp_cuda_graph",),)
    repeat = 2
    num_frames = 120
    warmup_frames = 30


class TeleopMuJoCo(_TeleopMuJoCoBenchmark):
    """Nightly teleop benchmark covering GPU and CPU solver backends."""


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastTeleopMuJoCo": FastTeleopMuJoCo,
        "TeleopMuJoCo": TeleopMuJoCo,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple (e.g., --bench A --bench B).",
    )
    args = parser.parse_known_args()[0]

    benchmarks = args.bench if args.bench is not None else benchmark_list.keys()
    for key in benchmarks:
        run_benchmark(benchmark_list[key])
