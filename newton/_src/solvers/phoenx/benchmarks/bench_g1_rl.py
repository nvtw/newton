# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX G1 RL throughput benchmark and nanoG1 comparison.

Measures the graph-captured PhoenX vector environment step path used by
``newton.rl.EnvG1PhoenX``. The optional nanoG1 comparison either consumes a
saved JSON result or invokes nanoG1's own Modal benchmark from the local
checkout.

Examples:
    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_rl
    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_rl --world-count 8192
    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_rl --run-nanog1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe

_NANOG1_RESULT_START = "=== ULTRA-BENCH RESULT ==="
_NANOG1_RESULT_END = "=== END RESULT ==="


def _default_steps_per_graph(sim_substeps: int) -> int:
    return 1 if int(sim_substeps) % 2 == 0 else 2


def benchmark_phoenx(
    *,
    world_count: int,
    sim_substeps: int,
    solver_iterations: int,
    velocity_iterations: int,
    parse_meshes: bool,
    measure_replays: int,
    warmup_steps: int,
    steps_per_graph: int | None,
    device: wp.context.Devicelike = None,
) -> dict[str, Any]:
    """Benchmark PhoenX G1 graph-captured environment stepping."""

    dev = wp.get_device(device)
    if not dev.is_cuda or not wp.is_mempool_enabled(dev):
        raise RuntimeError("PhoenX G1 RL benchmark requires CUDA with Warp mempool enabled")

    setup_t0 = time.perf_counter()
    cfg = rl.ConfigEnvG1PhoenX(
        world_count=int(world_count),
        sim_substeps=int(sim_substeps),
        solver_iterations=int(solver_iterations),
        velocity_iterations=int(velocity_iterations),
        parse_meshes=bool(parse_meshes),
    )
    env = rl.EnvG1PhoenX(cfg, device=dev)
    actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=env.device)
    graph_steps = int(steps_per_graph) if steps_per_graph is not None else _default_steps_per_graph(sim_substeps)
    if int(sim_substeps) * graph_steps % 2 != 0:
        raise ValueError("sim_substeps * steps_per_graph must be even for stable PhoenX state ping-pong replay")

    graph = rl.capture_env_steps(env, actions, steps_per_graph=graph_steps, warmup_steps=int(warmup_steps))
    wp.synchronize_device(dev)
    setup_seconds = time.perf_counter() - setup_t0

    t0 = time.perf_counter()
    for _ in range(int(measure_replays)):
        wp.capture_launch(graph)
    wp.synchronize_device(dev)
    elapsed = time.perf_counter() - t0

    env_steps = int(world_count) * int(measure_replays) * graph_steps
    physics_steps = env_steps * int(sim_substeps)
    env_steps_per_s = float(env_steps) / max(elapsed, 1.0e-12)
    physics_steps_per_s = float(physics_steps) / max(elapsed, 1.0e-12)
    return {
        "engine": "phoenx_g1_rl",
        "metric": "graph-captured environment-step throughput, no learner",
        "device": dev.name,
        "world_count": int(world_count),
        "obs_dim": rl.OBS_DIM_G1,
        "action_dim": rl.ACTION_DIM_G1,
        "sim_substeps": int(sim_substeps),
        "solver_iterations": int(solver_iterations),
        "velocity_iterations": int(velocity_iterations),
        "parse_meshes": bool(parse_meshes),
        "warmup_steps": int(warmup_steps),
        "measure_replays": int(measure_replays),
        "steps_per_graph": graph_steps,
        "setup_seconds": setup_seconds,
        "elapsed_s": elapsed,
        "env_steps_per_s": env_steps_per_s,
        "physics_steps_per_s": physics_steps_per_s,
    }


def _extract_nanog1_result(text: str) -> dict[str, Any]:
    start = text.find(_NANOG1_RESULT_START)
    end = text.find(_NANOG1_RESULT_END)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("nanoG1 benchmark output did not contain a JSON result block")
    blob = text[start + len(_NANOG1_RESULT_START) : end].strip()
    return json.loads(blob)


def _load_nanog1_result(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _extract_nanog1_result(text)


def _run_nanog1_benchmark(*, checkout: Path, config: str, timeout: int) -> dict[str, Any]:
    cmd = ["modal", "run", "bench/bench_nanog1.py", "--config", config]
    result = subprocess.run(
        cmd,
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=int(timeout),
        check=False,
    )
    if result.returncode != 0:
        tail = (result.stdout + "\n" + result.stderr)[-5000:]
        raise RuntimeError(f"nanoG1 benchmark failed with exit code {result.returncode}:\n{tail}")
    return _extract_nanog1_result(result.stdout)


def _compare(phoenx: dict[str, Any], nanog1: dict[str, Any] | None) -> dict[str, Any] | None:
    if nanog1 is None:
        return None
    nano_env = float(nanog1.get("peak_env_steps_per_s") or 0.0)
    nano_physics = float(nanog1.get("peak_physics_steps_per_s") or 0.0)
    return {
        "phoenx_over_nanog1_env_step_ratio": float(phoenx["env_steps_per_s"]) / nano_env if nano_env > 0.0 else None,
        "phoenx_over_nanog1_physics_step_ratio": (
            float(phoenx["physics_steps_per_s"]) / nano_physics if nano_physics > 0.0 else None
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=4096)
    parser.add_argument("--sim-substeps", type=int, default=5)
    parser.add_argument("--solver-iterations", type=int, default=2)
    parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    parser.add_argument("--parse-meshes", action="store_true")
    parser.add_argument("--measure-replays", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--steps-per-graph", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--nanog1-checkout", type=Path, default=Path("/home/twidmer/Documents/git/nanoG1"))
    parser.add_argument("--nanog1-config", choices=("production", "matched"), default="production")
    parser.add_argument("--nanog1-result-json", type=Path, default=None)
    parser.add_argument("--run-nanog1", action="store_true")
    parser.add_argument("--nanog1-timeout", type=int, default=3600)
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    phoenx = benchmark_phoenx(
        world_count=args.world_count,
        sim_substeps=args.sim_substeps,
        solver_iterations=args.solver_iterations,
        velocity_iterations=args.velocity_iterations,
        parse_meshes=args.parse_meshes,
        measure_replays=args.measure_replays,
        warmup_steps=args.warmup_steps,
        steps_per_graph=args.steps_per_graph,
        device=args.device,
    )

    nanog1 = None
    if args.nanog1_result_json is not None:
        nanog1 = _load_nanog1_result(args.nanog1_result_json)
    elif args.run_nanog1:
        nanog1 = _run_nanog1_benchmark(
            checkout=args.nanog1_checkout, config=args.nanog1_config, timeout=args.nanog1_timeout
        )

    output = {
        "phoenx": phoenx,
        "nanog1": nanog1,
        "comparison": _compare(phoenx, nanog1),
        "nanog1_reference_command": (
            f"cd {args.nanog1_checkout} && modal run bench/bench_nanog1.py --config {args.nanog1_config}"
        ),
    }
    print(json.dumps(output, indent=args.json_indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
