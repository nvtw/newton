# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure fixed or automatically tuned FlashSAC wall-to-quality on Ant."""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

from ..rl_training.ant import ConfigEnvAntPhoenX, EnvAntPhoenX, default_ant_flash_sac_config
from ..rl_training.flash_sac import BufferReplayFlashSAC, TrainerFlashSAC
from ..rl_training.flash_sac_autotune import ConfigFlashSACLRAutotune, ControllerFlashSACLRAutotune
from ..rl_training.flash_sac_autotune_evaluation import EvaluatorPairedFlashSAC


def _load_cudart() -> ctypes.CDLL:
    """Load the CUDA profiler control API."""

    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    raise RuntimeError("CUDA runtime library not found; profiling requires CUDA")


def _warm_replay(
    trainer: TrainerFlashSAC,
    env: EnvAntPhoenX,
    *,
    seed: int,
) -> BufferReplayFlashSAC:
    """Fill replay before graph capture without changing the measured interval."""

    replay = trainer.initialize_replay_buffer()
    replay.reserve_graph_buffers(env.world_count)
    obs = env.reset()
    pre_step_obs = wp.empty_like(obs)
    truncateds = wp.zeros(env.world_count, dtype=wp.float32, device=trainer.device)
    row_steps = (int(trainer.config.buffer_min_length) + env.world_count - 1) // env.world_count
    for step in range(row_steps + int(trainer.config.n_step) - 1):
        actions, _log_probs = trainer.act(obs, seed=seed + step)
        wp.copy(pre_step_obs, obs)
        next_obs, rewards, dones = env.step(actions)
        replay.add_batch_graph(
            pre_step_obs,
            actions,
            rewards,
            getattr(env, "step_terminateds", dones),
            getattr(env, "step_next_obs", next_obs),
            truncateds=getattr(env, "step_truncateds", truncateds),
        )
        obs = next_obs
    wp.synchronize_device(trainer.device)
    return replay


def _make_env_config(args: argparse.Namespace, *, world_count: int) -> ConfigEnvAntPhoenX:
    """Build one explicit Ant protocol shared by training and evaluation."""

    return ConfigEnvAntPhoenX(
        world_count=int(world_count),
        articulation_mode=str(args.articulation_mode),
        task_profile="mraksha",
        auto_reset=True,
    )


def _make_evaluator(
    sources: tuple[TrainerFlashSAC, TrainerFlashSAC],
    env_config: ConfigEnvAntPhoenX,
    args: argparse.Namespace,
) -> EvaluatorPairedFlashSAC:
    """Capture deterministic no-reset Ant evaluation."""

    eval_config = replace(
        env_config,
        world_count=int(args.evaluation_worlds),
        auto_reset=False,
        max_episode_steps=0,
    )
    envs = (EnvAntPhoenX(eval_config, device=args.device), EnvAntPhoenX(eval_config, device=args.device))

    def forward_velocity(env: EnvAntPhoenX, _rewards: wp.array[wp.float32]) -> wp.array[wp.float32]:
        return env.step_forward_velocities

    return EvaluatorPairedFlashSAC(
        sources,
        envs,
        horizon_steps=int(args.evaluation_horizon),
        seed=10_001 + int(args.seed) * 9973,
        metric_source=forward_velocity,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run one measured Ant seed and return its quality history."""

    device = wp.get_device(args.device)
    env_config = _make_env_config(args, world_count=int(args.world_count))
    config = default_ant_flash_sac_config()
    setup_start = time.perf_counter()
    env = EnvAntPhoenX(env_config, device=device)
    champion = TrainerFlashSAC(
        obs_dim=env.obs_dim,
        action_dim=env.policy_action_dim,
        config=config,
        device=device,
        seed=int(args.seed),
    )
    replay = _warm_replay(champion, env, seed=31 + int(args.seed) * 9973)
    controller = None
    if args.mode == "fixed":
        training = champion.capture_training_graph(
            env,
            replay,
            updates_per_step=2,
            interactions_per_graph=2,
            seed=31 + int(args.seed) * 9973,
            overlap=True,
        )
        sources = (champion, champion)
    else:
        controller = ControllerFlashSACLRAutotune.from_trainer(
            champion,
            rollout_world_count=env.world_count,
            config=ConfigFlashSACLRAutotune(
                evaluation_episodes=int(args.evaluation_worlds),
                initial_perturbation_factor=2.0,
                minimum_evidence_windows=2,
                promotion_windows=2,
                exploit_after_candidate=True,
                seed=int(args.seed),
            ),
        )
        training = controller.capture_overlap(
            env,
            replay,
            updates_per_step=2,
            interactions_per_launch=2,
            seed=31 + int(args.seed) * 9973,
            population_backend="parallel",
        )
        sources = training.evaluation_trainers()
    evaluator = _make_evaluator(sources, env_config, args)
    setup_seconds = time.perf_counter() - setup_start

    history: list[dict[str, object]] = []
    evaluation_seconds = 0.0
    consecutive_passes = 0
    run_start = time.perf_counter()
    completed_launches = 0
    cudart = _load_cudart() if args.cuda_profiler_api else None
    profiler_active = False
    try:
        if cudart is not None:
            if cudart.cudaProfilerStart() != 0:
                raise RuntimeError("cudaProfilerStart failed")
            profiler_active = True
        for launch in range(1, int(args.max_launches) + 1):
            training.launch()
            completed_launches = launch
            if launch % int(args.evaluation_interval) != 0:
                continue
            evaluation_start = time.perf_counter()
            training.synchronize()
            if controller is None:
                eval_sources = (champion, champion)
                evaluated_rates = [config.actor_lr, config.critic_lr, config.alpha_lr]
                action = "fixed"
            else:
                eval_sources = training.evaluation_trainers()
                evaluated_rates = controller.member_rates.tolist()
            scores = evaluator.evaluate(eval_sources)
            mean_velocity = float(np.mean(scores.champion_scores))
            passed = bool(
                scores.champion_finite
                and scores.champion_termination_rate <= float(args.maximum_termination_rate)
                and mean_velocity >= float(args.minimum_forward_velocity)
            )
            consecutive_passes = consecutive_passes + 1 if passed else 0
            if controller is not None and not controller.converged:
                decision = training.evaluate_paired(
                    scores.champion_scores,
                    scores.challenger_scores,
                    challenger_safe=scores.champion_finite and scores.challenger_finite,
                    champion_termination_rate=scores.champion_termination_rate,
                    challenger_termination_rate=scores.challenger_termination_rate,
                )
                action = decision.action
            elif controller is not None:
                action = "exploit"
            evaluation_seconds += time.perf_counter() - evaluation_start
            history.append(
                {
                    "launch": launch,
                    "transitions": launch * env.world_count * 2,
                    "champion_velocity": mean_velocity,
                    "challenger_velocity": float(np.mean(scores.challenger_scores)),
                    "champion_termination_rate": scores.champion_termination_rate,
                    "challenger_termination_rate": scores.challenger_termination_rate,
                    "passed": passed,
                    "consecutive_passes": consecutive_passes,
                    "action": action,
                    "evaluated_rates": evaluated_rates,
                    "next_rates": controller.member_rates.tolist() if controller is not None else evaluated_rates,
                }
            )
            if consecutive_passes >= 2:
                break
        training.synchronize()
    finally:
        if profiler_active:
            training.synchronize()
            if cudart.cudaProfilerStop() != 0:
                raise RuntimeError("cudaProfilerStop failed")
        training.close()
    total_seconds = time.perf_counter() - run_start
    training_seconds = total_seconds - evaluation_seconds
    transitions = completed_launches * env.world_count * 2
    result = {
        "mode": args.mode,
        "seed": int(args.seed),
        "world_count": env.world_count,
        "articulation_mode": args.articulation_mode,
        "setup_seconds": setup_seconds,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": total_seconds,
        "transitions": transitions,
        "training_transitions_per_second": transitions / max(training_seconds, 1.0e-12),
        "wall_transitions_per_second": transitions / max(total_seconds, 1.0e-12),
        "quality_gate": consecutive_passes >= 2,
        "history": history,
    }
    return result


def main() -> int:
    """Run the command-line benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed", "autotune"), default="autotune")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world-count", type=int, default=2048)
    parser.add_argument(
        "--articulation-mode",
        choices=("maximal", "maximal_projected", "maximal_articulated", "hybrid", "reduced"),
        default="reduced",
    )
    parser.add_argument("--max-launches", type=int, default=3000)
    parser.add_argument("--evaluation-interval", type=int, default=250)
    parser.add_argument("--evaluation-worlds", type=int, default=32)
    parser.add_argument("--evaluation-horizon", type=int, default=200)
    parser.add_argument("--minimum-forward-velocity", type=float, default=0.4)
    parser.add_argument("--maximum-termination-rate", type=float, default=0.06)
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="Bracket the measured launch loop for Nsight Systems capture.",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/phoenx_flash_sac_ant.json"))
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
