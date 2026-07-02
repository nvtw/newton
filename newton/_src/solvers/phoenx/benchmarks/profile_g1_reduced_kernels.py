# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded Nsight profile driver for reduced-coordinate PhoenX G1 physics.

The CUDA profiler API excludes environment construction, kernel compilation,
and graph capture. Only one to ten steady-state CUDA graph replays are visible
to Nsight, which keeps profiling time and report size bounded.

Examples:
    nsys profile --trace=cuda --capture-range=cudaProfilerApi \
        --capture-range-end=stop --cuda-graph-trace=node \
        --force-overwrite=true --output=/tmp/phoenx_g1_reduced \
        uv run --extra dev -m \
        newton._src.solvers.phoenx.benchmarks.profile_g1_reduced_kernels

    nsys stats --report cuda_gpu_kern_sum /tmp/phoenx_g1_reduced.nsys-rep
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time

import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe


def _load_cudart() -> ctypes.CDLL:
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    raise RuntimeError("CUDA runtime library not found; profiling requires CUDA")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=8192)
    parser.add_argument("--replays", type=int, choices=range(1, 11), default=5)
    parser.add_argument("--warmup-replays", type=int, default=2)
    parser.add_argument("--sim-substeps", type=int, default=5)
    parser.add_argument("--solver-iterations", type=int, default=2)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--prepare-refresh-stride", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("reduced G1 profiling requires CUDA with Warp mempool enabled")
    if args.warmup_replays < 0:
        raise ValueError("--warmup-replays must be non-negative")

    env = rl.EnvG1PhoenX(
        rl.ConfigEnvG1PhoenX(
            world_count=args.world_count,
            sim_substeps=args.sim_substeps,
            solver_iterations=args.solver_iterations,
            velocity_iterations=args.velocity_iterations,
            articulation_mode="reduced",
            actuation_model=g1_recipe.ACTUATION_MODEL,
            controlled_action_count=g1_recipe.CONTROLLED_ACTION_COUNT,
            parse_meshes=False,
            contact_geometry=g1_recipe.CONTACT_GEOMETRY,
            rigid_contact_max_per_world=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD,
            threads_per_world=g1_recipe.THREADS_PER_WORLD,
            multi_world_scheduler=g1_recipe.MULTI_WORLD_SCHEDULER,
            prepare_refresh_stride=args.prepare_refresh_stride,
        ),
        device=device,
    )
    actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)
    steps_per_graph = 1 if args.sim_substeps % 2 == 0 else 2
    graph = rl.capture_env_steps(env, actions, steps_per_graph=steps_per_graph, warmup_steps=4)
    for _ in range(args.warmup_replays):
        wp.capture_launch(graph)
    wp.synchronize_device(device)

    cudart = _load_cudart()
    if cudart.cudaProfilerStart() != 0:
        raise RuntimeError("cudaProfilerStart failed")
    start = time.perf_counter()
    for _ in range(args.replays):
        wp.capture_launch(graph)
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - start
    if cudart.cudaProfilerStop() != 0:
        raise RuntimeError("cudaProfilerStop failed")

    env_steps = args.world_count * args.replays * steps_per_graph
    print(
        json.dumps(
            {
                "articulation_mode": "reduced",
                "world_count": args.world_count,
                "sim_substeps": args.sim_substeps,
                "solver_iterations": args.solver_iterations,
                "velocity_iterations": args.velocity_iterations,
                "prepare_refresh_stride": args.prepare_refresh_stride,
                "graph_replays": args.replays,
                "steps_per_graph": steps_per_graph,
                "profiled_env_steps": env_steps,
                "elapsed_s": elapsed,
                "env_steps_per_s_under_profiler": env_steps / elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
