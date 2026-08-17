# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a bounded real-G1 FlashSAC learning-rate autotune experiment."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.rl_training.flash_sac import _allocate_flash_sac_batch
from newton._src.solvers.phoenx.rl_training.flash_sac_autotune import (
    ConfigFlashSACLRAutotune,
    ControllerFlashSACLRAutotune,
)
from newton._src.solvers.phoenx.rl_training.flash_sac_autotune_evaluation import EvaluatorPairedFlashSAC


@wp.kernel
def _root_command_velocity_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    q_stride: int,
    qd_stride: int,
    velocity: wp.array[wp.float32],
):
    world = wp.tid()
    q_base = world * q_stride
    qd_base = world * qd_stride
    rotation = wp.quat(
        joint_q[q_base + 3],
        joint_q[q_base + 4],
        joint_q[q_base + 5],
        joint_q[q_base + 6],
    )
    linear_world = wp.vec3(joint_qd[qd_base], joint_qd[qd_base + 1], joint_qd[qd_base + 2])
    velocity[world] = wp.quat_rotate_inv(rotation, linear_world)[0]


@wp.kernel
def _root_command_tracking_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    base_tracking: wp.array[wp.float32],
    q_stride: int,
    qd_stride: int,
    command_x: wp.float32,
    tracking: wp.array[wp.float32],
):
    world = wp.tid()
    q_base = world * q_stride
    qd_base = world * qd_stride
    rotation = wp.quat(
        joint_q[q_base + 3],
        joint_q[q_base + 4],
        joint_q[q_base + 5],
        joint_q[q_base + 6],
    )
    linear_world = wp.vec3(joint_qd[qd_base], joint_qd[qd_base + 1], joint_qd[qd_base + 2])
    command_velocity = wp.quat_rotate_inv(rotation, linear_world)[0]
    progress = wp.clamp(command_velocity / command_x, wp.float32(0.0), wp.float32(1.0))
    tracking[world] = progress * base_tracking[world]


def _warm_replay(trainer: rl.TrainerFlashSAC, env: rl.EnvG1PhoenX, seed: int) -> rl.BufferReplayFlashSAC:
    replay = trainer.initialize_replay_buffer()
    replay.reserve_graph_buffers(env.world_count)
    obs = env.reset()
    pre_step_obs = wp.empty_like(obs)
    truncateds = wp.zeros(env.world_count, dtype=wp.float32, device=trainer.device)
    row_steps = (int(trainer.config.buffer_min_length) + env.world_count - 1) // env.world_count
    warmup_steps = row_steps + int(trainer.config.n_step) - 1
    for step in range(warmup_steps):
        actions, _log_probs = trainer.act(obs, seed=seed + step)
        wp.copy(pre_step_obs, obs)
        next_obs, rewards, _dones = env.step(actions)
        replay.add_batch_graph(
            pre_step_obs,
            actions,
            rewards,
            env.step_terminateds,
            env.step_next_obs,
            truncateds=getattr(env, "step_truncateds", truncateds),
        )
        obs = next_obs
    wp.synchronize_device(trainer.device)
    return replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=2048)
    parser.add_argument("--base-lr", type=float, default=6.0e-4)
    parser.add_argument("--launches", type=int, default=4000)
    parser.add_argument("--evaluation-interval", type=int, default=400)
    parser.add_argument("--evaluation-episodes", type=int, default=32)
    parser.add_argument("--evaluation-horizon", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("/tmp/phoenx_flash_sac_autotune_g1.json"))
    parser.add_argument(
        "--finalization",
        choices=("best_confirmed", "live", "none"),
        default="best_confirmed",
    )
    args = parser.parse_args()
    device = wp.get_device("cuda:0")
    env_config = g1_recipe.isaaclab_flat_g1_env_config(
        world_count=args.world_count,
        command=(0.8, 0.0, 0.0),
        command_x_range=(0.8, 0.8),
        command_y_range=(0.0, 0.0),
        command_yaw_range=(0.0, 0.0),
        randomize_commands_on_reset=False,
        command_resample_steps=0,
    )
    config = g1_recipe.isaaclab_flat_g1_flash_sac_config(
        buffer_max_length=10_000_000,
        buffer_min_length=100_000,
        use_amp=True,
    )
    config = replace(config, actor_lr=args.base_lr, critic_lr=args.base_lr, alpha_lr=args.base_lr)
    setup_start = time.perf_counter()
    env = rl.EnvG1PhoenX(env_config, device=device)
    trainers = tuple(
        rl.TrainerFlashSAC(
            obs_dim=env.obs_dim,
            action_dim=env.policy_action_dim,
            config=config,
            device=device,
            seed=member,
        )
        for member in range(2)
    )
    controller = ControllerFlashSACLRAutotune(
        trainers,
        _allocate_flash_sac_batch(trainers[0]),
        rollout_world_count=args.world_count,
        config=ConfigFlashSACLRAutotune(
            evaluation_episodes=args.evaluation_episodes,
            promotion_windows=2,
            convergence_windows=100,
        ),
    )
    replay = _warm_replay(trainers[0], env, seed=31)
    training = controller.capture_overlap(
        env,
        replay,
        updates_per_step=2,
        interactions_per_launch=2,
        seed=31,
        population_backend="parallel",
    )
    eval_config = replace(env_config, world_count=args.evaluation_episodes, auto_reset=False, max_episode_steps=0)
    eval_envs = (rl.EnvG1PhoenX(eval_config, device=device), rl.EnvG1PhoenX(eval_config, device=device))
    tracking_metrics = tuple(wp.empty(args.evaluation_episodes, dtype=wp.float32, device=device) for _ in range(2))

    def command_tracking_metric(heldout: rl.EnvG1PhoenX, _rewards: wp.array[wp.float32]) -> wp.array[wp.float32]:
        member = 0 if heldout is eval_envs[0] else 1
        wp.launch(
            _root_command_tracking_kernel,
            dim=args.evaluation_episodes,
            inputs=[
                heldout.state_0.joint_q,
                heldout.state_0.joint_qd,
                heldout.step_successes,
                int(heldout.model.joint_coord_count) // args.evaluation_episodes,
                int(heldout.model.joint_dof_count) // args.evaluation_episodes,
                0.8,
            ],
            outputs=[tracking_metrics[member]],
            device=device,
        )
        return tracking_metrics[member]

    evaluator = EvaluatorPairedFlashSAC(
        training.trainers,
        eval_envs,
        horizon_steps=args.evaluation_horizon,
        seed=1101,
        metric_source=command_tracking_metric,
    )
    setup_seconds = time.perf_counter() - setup_start
    history = []
    evaluation_seconds = 0.0
    run_start = time.perf_counter()
    for launch in range(1, args.launches + 1):
        training.launch()
        if launch % args.evaluation_interval == 0:
            evaluation_start = time.perf_counter()
            training.synchronize()
            challenger_fallback_fraction = training.challenger_fallback_fraction()
            scores = evaluator.evaluate(training.trainers)
            evaluated_rates = controller.member_rates.tolist()
            decision = training.evaluate_paired(
                scores.champion_scores,
                scores.challenger_scores,
                challenger_safe=scores.champion_finite and scores.challenger_finite,
                champion_termination_rate=scores.champion_termination_rate,
                challenger_termination_rate=scores.challenger_termination_rate,
            )
            history.append(
                {
                    "launch": launch,
                    "champion_success": float(np.mean(scores.champion_scores)),
                    "challenger_success": float(np.mean(scores.challenger_scores)),
                    "champion_finite": scores.champion_finite,
                    "challenger_finite": scores.challenger_finite,
                    "champion_termination_rate": scores.champion_termination_rate,
                    "challenger_termination_rate": scores.challenger_termination_rate,
                    "challenger_action_fallback_fraction": challenger_fallback_fraction,
                    "challenger_relative_safe": scores.champion_finite
                    and scores.challenger_finite
                    and scores.challenger_termination_rate
                    <= scores.champion_termination_rate + controller.config.termination_rate_margin,
                    "action": decision.action,
                    "evaluated_rates": evaluated_rates,
                    "next_rates": controller.member_rates.tolist(),
                }
            )
            evaluation_seconds += time.perf_counter() - evaluation_start
    final_evaluation_start = time.perf_counter()
    training.sync_controller_state()
    if args.finalization == "live":
        live_scores = evaluator.evaluate(controller.trainers)
        controller.finalize_best(
            policy="live",
            live_score=float(np.mean(live_scores.champion_scores)),
            live_termination_rate=live_scores.champion_termination_rate,
        )
    else:
        controller.finalize_best(policy=args.finalization)
    final_trainers = (
        (controller.single_trainer, controller.single_trainer) if args.finalization != "none" else controller.trainers
    )
    final_scores = evaluator.evaluate(final_trainers)
    velocity = wp.empty(args.evaluation_episodes, dtype=wp.float32, device=device)
    quality_env = eval_envs[0]
    wp.launch(
        _root_command_velocity_kernel,
        dim=args.evaluation_episodes,
        inputs=[
            quality_env.state_0.joint_q,
            quality_env.state_0.joint_qd,
            int(quality_env.model.joint_coord_count) // args.evaluation_episodes,
            int(quality_env.model.joint_dof_count) // args.evaluation_episodes,
        ],
        outputs=[velocity],
        device=device,
    )
    final_velocities = velocity.numpy()
    final_velocity = float(np.mean(final_velocities))
    evaluation_seconds += time.perf_counter() - final_evaluation_start
    total_run_seconds = time.perf_counter() - run_start
    result = {
        "setup_seconds": setup_seconds,
        "total_run_seconds": total_run_seconds,
        "evaluation_seconds": evaluation_seconds,
        "training_only_seconds": total_run_seconds - evaluation_seconds,
        "transitions_per_second": args.world_count * 2 * args.launches / total_run_seconds,
        "initial_rates": [args.base_lr, args.base_lr, args.base_lr],
        "finalization": args.finalization,
        "final_rates": controller.member_rates.tolist(),
        "final_champion_success": float(np.mean(final_scores.champion_scores)),
        "final_challenger_success": float(np.mean(final_scores.challenger_scores)),
        "final_champion_termination_rate": final_scores.champion_termination_rate,
        "final_challenger_termination_rate": final_scores.challenger_termination_rate,
        "final_zero_fall_gate": bool(final_scores.champion_finite and final_scores.champion_termination_rate == 0.0),
        "final_tracking_gate": bool(np.mean(final_scores.champion_scores) >= 0.30),
        "final_velocity_gate": bool(final_velocity >= 0.4),
        "final_quality_gate": bool(
            final_scores.champion_finite
            and final_scores.champion_termination_rate == 0.0
            and np.mean(final_scores.champion_scores) >= 0.30
            and final_velocity >= 0.4
        ),
        "final_command_aligned_velocity": final_velocity,
        "history": history,
        "best_score": controller.best_score,
        "best_termination_rate": controller.best_termination_rate,
        "best_rates": controller.best_rates.tolist(),
    }
    training.close()
    output = json.dumps(result, indent=2, sort_keys=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
