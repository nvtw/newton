# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded Nsight profile driver for captured PhoenX FlashSAC training.

Examples:
    nsys profile --trace=cuda --capture-range=cudaProfilerApi \
        --capture-range-end=stop --cuda-graph-trace=node \
        --force-overwrite=true --output=/tmp/flash_full \
        uv run -m newton._src.solvers.phoenx.benchmarks.profile_flash_sac_training

    nsys stats --report cuda_gpu_kern_sum /tmp/flash_full.nsys-rep
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time

import numpy as np
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


def _config(*, minimum_size: int, use_amp: bool) -> rl.ConfigFlashSAC:
    return rl.ConfigFlashSAC(
        buffer_max_length=max(minimum_size * 4, 16_384),
        buffer_min_length=minimum_size,
        sample_batch_size=2048,
        n_step=3,
        use_amp=use_amp,
    )


def _profile_learner(
    device: wp.Device, replays: int, warmup_replays: int, use_amp: bool, action_dim: int
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(17)
    trainer = rl.TrainerFlashSAC(
        obs_dim=98, action_dim=action_dim, config=_config(minimum_size=2048, use_amp=use_amp), device=device, seed=19
    )
    batch = rl.BatchSAC(
        obs=wp.array(rng.normal(size=(2048, 98)).astype(np.float32), device=device),
        actions=wp.array(rng.uniform(-1.0, 1.0, size=(2048, action_dim)).astype(np.float32), device=device),
        rewards=wp.array(rng.normal(size=2048).astype(np.float32), device=device),
        dones=wp.zeros(2048, dtype=wp.float32, device=device),
        next_obs=wp.array(rng.normal(size=(2048, 98)).astype(np.float32), device=device),
    )
    graph = trainer.capture_update_graph(batch, seed=23)
    for _ in range(warmup_replays):
        graph.launch()
    wp.synchronize_device(device)
    cudart = _load_cudart()
    if cudart.cudaProfilerStart() != 0:
        raise RuntimeError("cudaProfilerStart failed")
    start = time.perf_counter()
    for _ in range(replays):
        graph.launch()
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - start
    if cudart.cudaProfilerStop() != 0:
        raise RuntimeError("cudaProfilerStop failed")
    return {
        "mode": "learner",
        "use_amp": use_amp,
        "action_dim": action_dim,
        "replays": replays,
        "elapsed_seconds": elapsed,
        "milliseconds_per_update": elapsed * 1000.0 / replays,
    }


def _profile_full(
    device: wp.Device,
    replays: int,
    warmup_replays: int,
    worlds: int,
    use_amp: bool,
    overlap: bool = False,
    full_action: bool = False,
) -> dict[str, float | int | str]:
    setup_start = time.perf_counter()
    env_config = (
        g1_recipe.isaaclab_flat_g1_env_config(world_count=worlds)
        if full_action
        else g1_recipe.default_g1_env_config(world_count=worlds)
    )
    config = (
        g1_recipe.isaaclab_flat_g1_flash_sac_config(
            buffer_max_length=max(worlds * 4, 16_384), buffer_min_length=worlds, use_amp=use_amp
        )
        if full_action
        else _config(minimum_size=worlds, use_amp=use_amp)
    )
    env = rl.EnvG1PhoenX(env_config, device=device)
    trainer = rl.TrainerFlashSAC(
        obs_dim=env.obs_dim,
        action_dim=env.policy_action_dim,
        config=config,
        device=device,
        seed=29,
    )
    graph = trainer.prepare_training_graph(
        env,
        updates_per_step=2,
        interactions_per_graph=2,
        seed=31,
        overlap=overlap,
    )
    setup_seconds = time.perf_counter() - setup_start
    for _ in range(warmup_replays):
        graph.launch()
    wp.synchronize_device(device)
    cudart = _load_cudart()
    if cudart.cudaProfilerStart() != 0:
        raise RuntimeError("cudaProfilerStart failed")
    start = time.perf_counter()
    for _ in range(replays):
        graph.launch()
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - start
    if cudart.cudaProfilerStop() != 0:
        raise RuntimeError("cudaProfilerStop failed")
    transitions = worlds * 2 * replays
    return {
        "mode": "full",
        "use_amp": use_amp,
        "overlap": overlap,
        "full_action": full_action,
        "setup_seconds": setup_seconds,
        "world_count": worlds,
        "replays": replays,
        "elapsed_seconds": elapsed,
        "milliseconds_per_cadence": elapsed * 1000.0 / replays,
        "transitions_per_second": transitions / elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("learner", "full"), default="full")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--warmup-replays", type=int, default=5)
    parser.add_argument("--world-count", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true", help="Use FP16 FlashSAC contractions")
    parser.add_argument("--action-dim", type=int, default=29, help="Policy action width for learner-only profiling")
    parser.add_argument("--overlap", action="store_true", help="Overlap rollout and learner graphs")
    parser.add_argument("--full-action", action="store_true", help="Use the tuned IsaacLab-flat full-action G1 recipe")
    args = parser.parse_args()
    if args.replays <= 0 or args.warmup_replays < 0:
        parser.error("replays must be positive and warmup-replays non-negative")
    if args.action_dim <= 0:
        parser.error("action-dim must be positive")
    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("FlashSAC profiling requires CUDA with Warp memory pools enabled")
    if args.mode == "learner":
        result = _profile_learner(device, args.replays, args.warmup_replays, args.amp, args.action_dim)
    else:
        result = _profile_full(
            device, args.replays, args.warmup_replays, args.world_count, args.amp, args.overlap, args.full_action
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
