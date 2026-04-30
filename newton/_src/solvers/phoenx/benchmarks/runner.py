# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warmup + CUDA-graph-capture + measure harness for the PhoenX
benchmarks.

Kept small on purpose: a single :func:`run_one` call takes a built
scene, advances it through ``warmup_frames`` to compile kernels and
fill caches, captures the per-frame kernel sequence into a CUDA
graph, replays it for ``measure_frames``, and reports wall-clock
throughput. No Nsight, no per-kernel breakdown -- the minimum
viable comparison between PhoenX and MuJoCo Warp.
"""

from __future__ import annotations

import dataclasses
import gc
import time
from collections.abc import Callable

import warp as wp


@dataclasses.dataclass
class SceneHandle:
    """Everything a benchmark run needs to advance one simulation frame.

    Scenarios return an instance of this. The driver only ever calls
    :attr:`simulate_one_frame`; ``setup_bytes`` is the GPU-memory
    delta between before the scene was built and when it was handed
    back, used for the dashboard's memory chart.
    """

    name: str
    solver_name: str
    num_worlds: int
    substeps: int
    solver_iterations: int
    simulate_one_frame: Callable[[], None]
    #: Bytes of GPU memory the scene allocated during setup. Computed
    #: by :func:`run_one` using ``wp.get_device().total_memory -
    #: wp.get_device().free_memory`` bookends; approximate (other
    #: processes on the GPU perturb it) but close enough to catch
    #: allocation regressions.
    setup_bytes: int = 0


def _gpu_used_bytes() -> int:
    """Return current GPU used memory in bytes via Warp's context.

    Warp's :class:`wp.context.Device` exposes ``total_memory`` and
    ``free_memory`` on CUDA devices. On CPU devices both are 0, so
    the returned value is meaningless and the caller should gate on
    ``device.is_cuda``.
    """
    device = wp.get_device()
    if not device.is_cuda:
        return 0
    return int(device.total_memory - device.free_memory)


def run_one(
    handle: SceneHandle,
    *,
    warmup_frames: int = 16,
    measure_frames: int = 64,
) -> dict:
    """Advance ``handle`` for ``warmup_frames`` (discarded), capture
    one frame into a CUDA graph, replay for ``measure_frames`` with
    the wall-clock timer gated on a single end-of-loop
    :func:`wp.synchronize_device`.

    Returns a dict matching the ``points.jsonl`` schema the
    dashboard expects (subset of Dylan Turpin's nightly schema):
    ``env_fps``, ``ms_per_step``, ``elapsed_s``, ``gpu_used_gb``,
    plus ``ok`` / ``error`` sentinels for the driver to emit a row
    even on failure.
    """
    device = wp.get_device()

    # Warmup: kernel JIT, cudaMalloc paths, contact-sorter internal
    # buffers. Matches Dylan's 16 / 64 default.
    for _ in range(warmup_frames):
        handle.simulate_one_frame()
    wp.synchronize_device()

    # Capture one frame into a CUDA graph so the measurement loop is
    # a tight ``capture_launch`` without Python overhead per step.
    # Fall back to eager stepping on CPU (where ScopedCapture is a
    # no-op) but warn -- CPU timings are noisy.
    graph = None
    if device.is_cuda:
        with wp.ScopedCapture() as capture:
            handle.simulate_one_frame()
        graph = capture.graph
        wp.synchronize_device()

    gpu_total_gb = float(device.total_memory) / (1024**3) if device.is_cuda else 0.0

    # Steady-state GPU memory sample. Taken AFTER the capture so any
    # scratch allocated during capture (contact ingest scratch,
    # graph resources) is included.
    gpu_used_gb = float(_gpu_used_bytes()) / (1024**3) if device.is_cuda else 0.0

    # Measurement loop: wall-clock around ``measure_frames`` replays
    # with a single sync at the end. Including more syncs serialises
    # CPU/GPU and inflates timings; one final sync captures the last
    # kernel's completion and is enough.
    t0 = time.perf_counter()
    if graph is not None:
        for _ in range(measure_frames):
            wp.capture_launch(graph)
    else:
        for _ in range(measure_frames):
            handle.simulate_one_frame()
    wp.synchronize_device()
    elapsed_s = time.perf_counter() - t0

    # env_fps is frames/second scaled by the parallel world count
    # (Dylan's convention: "how many env steps did we produce per
    # second"). ms_per_step is the inverse, in milliseconds per
    # single-world step, easier to eyeball for regressions.
    total_env_steps = handle.num_worlds * measure_frames
    env_fps = float(total_env_steps) / elapsed_s
    ms_per_step = elapsed_s * 1000.0 / float(total_env_steps)

    return {
        "scenario": handle.name,
        "solver": handle.solver_name,
        "num_worlds": handle.num_worlds,
        "substeps": handle.substeps,
        "solver_iterations": handle.solver_iterations,
        "warmup_frames": warmup_frames,
        "measure_frames": measure_frames,
        "elapsed_s": elapsed_s,
        "env_fps": env_fps,
        "ms_per_step": ms_per_step,
        "gpu_used_gb": gpu_used_gb,
        "gpu_total_gb": gpu_total_gb,
        "setup_gb": float(handle.setup_bytes) / (1024**3),
        "ok": True,
        "error": None,
    }


def reset_gpu_between_runs() -> None:
    """Drop Python-side references to any scene state + trigger a
    garbage collection so the next :func:`run_one` starts with a
    fresh heap. Doesn't call :meth:`wp.synchronize_device` -- caller
    should do that explicitly after clearing references.
    """
    gc.collect()
