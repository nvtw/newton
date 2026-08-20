# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare production-shaped FlashSAC P1 and LR-autotune overlap throughput."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time

import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.rl_training.flash_sac import _allocate_flash_sac_batch
from newton._src.solvers.phoenx.rl_training.flash_sac_autotune import (
    ConfigFlashSACLRAutotune,
    ControllerFlashSACLRAutotune,
)
from newton._src.solvers.phoenx.rl_training.flash_sac_autotune_parallel import (
    capture_lr_autotune_parallel_overlap,
)


def _config(worlds: int) -> rl.ConfigFlashSAC:
    return g1_recipe.isaaclab_flat_g1_flash_sac_config(
        buffer_max_length=max(worlds * 4, 16_384),
        buffer_min_length=worlds,
        use_amp=True,
    )


def _warm_replay(
    trainer: rl.TrainerFlashSAC,
    env: rl.EnvG1PhoenX,
    *,
    seed: int,
) -> rl.BufferReplayFlashSAC:
    replay = trainer.initialize_replay_buffer()
    replay.reserve_graph_buffers(env.world_count)
    obs = env.reset()
    pre_step_obs = wp.empty_like(obs)
    zero_truncateds = wp.zeros(env.world_count, dtype=wp.float32, device=trainer.device)
    for step in range(int(trainer.config.n_step)):
        actions, _log_probs = trainer.act(obs, seed=seed + step)
        wp.copy(pre_step_obs, obs)
        next_obs, rewards, dones = env.step(actions)
        replay.add_batch_graph(
            pre_step_obs,
            actions,
            rewards,
            getattr(env, "step_terminateds", dones),
            getattr(env, "step_next_obs", next_obs),
            truncateds=getattr(env, "step_truncateds", zero_truncateds),
        )
        obs = next_obs
    wp.synchronize_device(trainer.device)
    if not replay.can_sample():
        raise RuntimeError("production benchmark replay warmup did not reach its minimum size")
    return replay


def _median_throughput(
    graph: object,
    device: wp.Device,
    *,
    worlds: int,
    warmups: int,
    trials: int,
    launches: int,
) -> dict[str, float]:
    for _ in range(warmups):
        graph.launch()
    graph.synchronize()
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        for _launch in range(launches):
            graph.launch()
        graph.synchronize()
        samples.append(worlds * 2 * launches / (time.perf_counter() - start))
    return {
        "median_transitions_per_second": statistics.median(samples),
        "minimum_transitions_per_second": min(samples),
        "maximum_transitions_per_second": max(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--launches", type=int, default=5)
    args = parser.parse_args()
    device = wp.get_device("cuda:0")
    worlds = int(args.world_count)
    env_config = g1_recipe.isaaclab_flat_g1_env_config(world_count=worlds)
    config = _config(worlds)
    result: dict[str, object] = {
        "world_count": worlds,
        "action_dim": 29,
        "batch_size": int(config.sample_batch_size),
        "interactions_per_launch": 2,
        "updates_per_launch": 4,
        "amp": bool(config.use_amp),
    }

    setup_start = time.perf_counter()
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
        overlap=True,
    )
    result["normal_p1_setup_seconds"] = time.perf_counter() - setup_start
    result["normal_p1"] = _median_throughput(
        graph,
        device,
        worlds=worlds,
        warmups=args.warmups,
        trials=args.trials,
        launches=args.launches,
    )
    graph.close()
    del graph, trainer, env
    gc.collect()
    wp.synchronize_device(device)

    setup_start = time.perf_counter()
    env = rl.EnvG1PhoenX(env_config, device=device)
    trainers = tuple(
        rl.TrainerFlashSAC(
            obs_dim=env.obs_dim,
            action_dim=env.policy_action_dim,
            config=config,
            device=device,
            seed=41 + member,
        )
        for member in range(2)
    )
    batch = _allocate_flash_sac_batch(trainers[0])
    controller = ControllerFlashSACLRAutotune(
        trainers,
        batch,
        rollout_world_count=worlds,
        config=ConfigFlashSACLRAutotune(evaluation_episodes=8),
    )
    replay = _warm_replay(trainers[0], env, seed=47)
    graph = controller.capture_overlap(
        env,
        replay,
        updates_per_step=2,
        interactions_per_launch=2,
        seed=53,
        population_backend="fused",
    )
    result["autotune_shared_setup_seconds"] = time.perf_counter() - setup_start
    result["autotune_p2"] = _median_throughput(
        graph,
        device,
        worlds=worlds,
        warmups=args.warmups,
        trials=args.trials,
        launches=args.launches,
    )
    graph.close()
    parallel_setup_start = time.perf_counter()
    graph = capture_lr_autotune_parallel_overlap(
        controller,
        env,
        replay,
        updates_per_step=2,
        interactions_per_launch=2,
        seed=59,
    )
    result["autotune_parallel_setup_seconds"] = time.perf_counter() - parallel_setup_start
    result["autotune_parallel_p2"] = _median_throughput(
        graph,
        device,
        worlds=worlds,
        warmups=args.warmups,
        trials=args.trials,
        launches=args.launches,
    )
    graph.sync_controller_state()
    switch_start = time.perf_counter()
    controller._converge_to_single()
    graph.synchronize()
    graph.sync_from_controller_state()
    result["autotune_convergence_switch_seconds"] = time.perf_counter() - switch_start
    result["autotune_converged_p1"] = _median_throughput(
        graph,
        device,
        worlds=worlds,
        warmups=args.warmups,
        trials=args.trials,
        launches=args.launches,
    )
    reopen_start = time.perf_counter()
    graph.reopen_search()
    result["autotune_reopen_switch_seconds"] = time.perf_counter() - reopen_start
    result["autotune_reopened_p2"] = _median_throughput(
        graph,
        device,
        worlds=worlds,
        warmups=args.warmups,
        trials=args.trials,
        launches=args.launches,
    )
    graph.close()
    del graph, controller, trainers, replay, env, batch
    gc.collect()
    wp.synchronize_device(device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
