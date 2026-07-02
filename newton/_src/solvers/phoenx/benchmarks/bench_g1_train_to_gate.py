# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure PhoenX G1 PPO train-to-gate progress.

This benchmark mirrors nanoG1's user-visible lifecycle: train from scratch,
write a checkpoint, reload it, and run the frozen G1 walking quality gate. It can
run a short smoke or continue chunks until the policy passes the gate.

Examples:
    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_train_to_gate \
        --max-iterations 2 --chunk-iterations 1 --execution-mode graph_leapfrog \
        --battery-steps 1 --diagnostic-steps 1 --diagnostic-world-count 1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.benchmarks.bench_g1_train import (
    _g1_ppo_config,
    _parse_hidden_layers,
    _parse_int_or_auto,
)
from newton._src.solvers.phoenx.rl_training import g1_recipe

_NANOG1_WALK_SAMPLES = 75_000_000
_NANOG1_WALK_SECONDS = 58.9
_NANOG1_WALK_SPS = 1_276_000.0
_NANOG1_REFERENCE = "nanoG1 RESULTS.md time-to-walk table"


def _format_checkpoint_template(path: str, iteration: int) -> Path:
    if "{iteration" in path:
        return Path(path.format(iteration=int(iteration)))
    return Path(path)


def _make_env_config(args: argparse.Namespace, *, world_count: int | None = None) -> rl.ConfigEnvG1PhoenX:
    return rl.ConfigEnvG1PhoenX(
        world_count=int(args.world_count if world_count is None else world_count),
        command=(float(args.command_x), float(args.command_y), float(args.command_yaw)),
        sim_substeps=int(args.sim_substeps),
        solver_iterations=int(args.solver_iterations),
        velocity_iterations=int(args.velocity_iterations),
        articulation_mode=str(getattr(args, "articulation_mode", "reduced")),
        actuation_model=str(getattr(args, "actuation_model", g1_recipe.ACTUATION_MODEL)),
        action_scale=float(args.action_scale),
        controlled_action_count=int(args.controlled_action_count),
        observation_mode=str(args.observation_mode),
        reward_mode=str(args.reward_mode),
        w_alive=float(args.w_alive),
        w_track_lin=float(args.w_track_lin),
        w_track_ang=float(args.w_track_ang),
        w_command_progress=float(args.w_command_progress),
        w_lin_vel_z=float(args.w_lin_vel_z),
        w_ang_vel_xy=float(args.w_ang_vel_xy),
        w_orientation=float(args.w_orientation),
        w_torque=float(args.w_torque),
        w_action_rate=float(args.w_action_rate),
        w_sparse_command_success=float(args.w_sparse_command_success),
        w_target_progress=float(args.w_target_progress),
        sparse_command_velocity_tolerance=float(args.sparse_command_velocity_tolerance),
        sparse_command_yaw_tolerance=float(args.sparse_command_yaw_tolerance),
        sparse_target_position=(float(args.target_x), float(args.target_y)),
        sparse_target_radius=float(args.sparse_target_radius),
        sparse_target_success_upright_cos=float(args.sparse_target_success_upright_cos),
        sparse_target_success_min_base_height=float(args.sparse_target_success_min_base_height),
        sparse_target_success_max_base_height=float(args.sparse_target_success_max_base_height),
        w_mechanical_power=float(args.w_mechanical_power),
        w_gait_contact=float(args.w_gait_contact),
        w_gait_swing=float(args.w_gait_swing),
        w_gait_swing_contact=float(args.w_gait_swing_contact),
        w_gait_hip=float(args.w_gait_hip),
        w_base_height=float(args.w_base_height),
        w_feet_air_time=float(args.w_feet_air_time),
        feet_air_time_threshold=float(args.feet_air_time_threshold),
        w_feet_slide=float(args.w_feet_slide),
        w_joint_deviation_hip=float(args.w_joint_deviation_hip),
        w_joint_deviation_waist=float(args.w_joint_deviation_waist),
        w_joint_deviation_upper=float(args.w_joint_deviation_upper),
        w_joint_acc_legs=float(args.w_joint_acc_legs),
        w_joint_pos_limit_ankle=float(args.w_joint_pos_limit_ankle),
        parse_meshes=bool(args.parse_meshes),
        contact_geometry=str(getattr(args, "contact_geometry", g1_recipe.CONTACT_GEOMETRY)),
        ground_friction=float(args.ground_friction),
        foot_box_xy_scale=float(args.foot_box_xy_scale),
        rigid_contact_max_per_world=int(args.rigid_contact_max_per_world),
        threads_per_world=args.threads_per_world,
        multi_world_scheduler=str(args.multi_world_scheduler),
        prepare_refresh_stride=args.prepare_refresh_stride,
    )


def _make_ppo_config(args: argparse.Namespace) -> rl.ConfigPPO:
    return _g1_ppo_config(
        args.train_epochs,
        args.actor_lr,
        args.critic_lr,
        args.anneal_lr,
        args.lr_anneal_timesteps,
        args.min_lr_ratio,
        args.mirror_loss_coeff,
        args.minibatch_size,
        args.replay_ratio,
        args.priority_alpha,
        args.priority_beta,
        not args.no_manual_actor_backward,
        not args.no_manual_critic_backward,
        args.policy_network,
        args.manual_mlp_weight_grad_dtype,
        args.manual_mlp_forward_dtype,
        args.vtrace_rho_clip,
        args.vtrace_c_clip,
        args.reward_clip,
        args.max_grad_norm,
        args.value_loss_coeff,
        args.value_clip_range,
        args.optimizer,
        args.optimizer_eps,
        args.optimizer_weight_decay,
        args.muon_momentum,
    )


def _samples_for_iteration(args: argparse.Namespace, iteration: int) -> int:
    return int(iteration) * int(args.world_count) * int(args.rollout_steps)


def _evaluate_checkpoint(
    checkpoint_path: str | Path,
    args: argparse.Namespace,
    *,
    device: wp.context.Device,
) -> rl.ResultEvaluateG1GatePPO:
    reloaded = rl.load_ppo_checkpoint(checkpoint_path, device=device)
    return rl.evaluate_g1_gate_ppo(
        reloaded,
        rl.ConfigEvaluateG1GatePPO(
            env_config=_make_env_config(args),
            battery_steps=int(args.battery_steps),
            seeds_per_command=int(args.seeds_per_command),
            diagnostic_steps=int(args.diagnostic_steps),
            diagnostic_world_count=int(args.diagnostic_world_count),
            device=device,
            deterministic=not bool(args.stochastic_gate),
            seed=int(args.gate_seed),
            max_battery_falls=int(args.max_battery_falls),
            min_battery_perf=float(args.min_battery_perf),
            max_action_jerk_rms=float(args.max_action_jerk_rms),
            max_ang_vel_xy_rms=float(args.max_ang_vel_xy_rms),
            max_yaw_rate_rms=float(args.max_yaw_rate_rms),
            max_leg_qvel_rms=float(args.max_leg_qvel_rms),
        ),
    )


def benchmark_train_to_gate(args: argparse.Namespace) -> dict[str, Any]:
    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("PhoenX G1 train-to-gate benchmark requires CUDA with Warp mempool enabled")
    if args.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if args.chunk_iterations <= 0:
        raise ValueError("chunk_iterations must be positive")
    if args.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if args.target_samples <= 0:
        raise ValueError("target_samples must be positive")

    checkpoint_template = args.checkpoint_path or "/tmp/phoenx_g1_gate_{iteration}.npz"
    env_config = _make_env_config(args)
    ppo_config = _make_ppo_config(args)
    command_curriculum_start = float(getattr(args, "command_curriculum_start", g1_recipe.COMMAND_CURRICULUM_START))
    command_curriculum_samples = int(getattr(args, "command_curriculum_samples", g1_recipe.COMMAND_CURRICULUM_SAMPLES))
    angular_fine_tune_start_samples = int(args.angular_fine_tune_start_samples)
    angular_fine_tune_iteration = (
        0
        if angular_fine_tune_start_samples <= 0
        else int(math.ceil(angular_fine_tune_start_samples / float(int(args.world_count) * int(args.rollout_steps))))
    )

    train_seconds = 0.0
    gate_seconds = 0.0
    resume_checkpoint: str | None = str(args.resume_checkpoint) if args.resume_checkpoint is not None else None
    completed_iterations = 0
    if resume_checkpoint is not None:
        completed_iterations = int(
            rl.load_ppo_checkpoint(resume_checkpoint, config=ppo_config, device=device).iteration
        )
        if not bool(args.evaluate_only) and completed_iterations >= int(args.max_iterations):
            raise ValueError("max_iterations must exceed the resumed checkpoint iteration")
    if bool(args.evaluate_only) and resume_checkpoint is None:
        raise ValueError("evaluate_only requires resume_checkpoint")
    start_iterations = int(completed_iterations)
    start_samples = _samples_for_iteration(args, start_iterations)
    gate_history: list[dict[str, Any]] = []
    train_history: list[dict[str, Any]] = []
    first_pass: dict[str, Any] | None = None
    total_t0 = time.perf_counter()

    if bool(args.evaluate_only):
        gate_t0 = time.perf_counter()
        gate_result = _evaluate_checkpoint(resume_checkpoint, args, device=device)
        gate_seconds += time.perf_counter() - gate_t0
        gate_entry = {
            "iteration": int(completed_iterations),
            "samples": int(_samples_for_iteration(args, completed_iterations)),
            "checkpoint": str(resume_checkpoint),
            "stats": asdict(gate_result.stats),
        }
        gate_history.append(gate_entry)
        if gate_result.stats.pass_gate:
            first_pass = gate_entry

    while not bool(args.evaluate_only) and completed_iterations < int(args.max_iterations):
        chunk_iterations = min(int(args.chunk_iterations), int(args.max_iterations) - completed_iterations)
        fine_tune_active = angular_fine_tune_iteration > 0 and completed_iterations >= angular_fine_tune_iteration
        if (
            angular_fine_tune_iteration > completed_iterations
            and completed_iterations + chunk_iterations > angular_fine_tune_iteration
        ):
            chunk_iterations = angular_fine_tune_iteration - completed_iterations
        active_env_config = env_config
        active_ppo_config = ppo_config
        if fine_tune_active:
            active_env_config = replace(env_config, w_ang_vel_xy=float(args.angular_fine_tune_w_ang_vel_xy))
            active_ppo_config = replace(
                ppo_config,
                lr_anneal_timesteps=int(args.angular_fine_tune_lr_anneal_timesteps),
            )
        chunk_t0 = time.perf_counter()
        result = rl.train_g1_ppo(
            rl.ConfigTrainG1PPO(
                iterations=chunk_iterations,
                rollout_steps=int(args.rollout_steps),
                hidden_layers=tuple(args.hidden_layers),
                activation=str(args.activation),
                env_config=active_env_config,
                ppo_config=active_ppo_config,
                device=device,
                seed=int(args.seed),
                log_interval=0,
                randomize_commands=not bool(args.no_command_randomization),
                command_x_range=tuple(float(v) for v in args.command_x_range),
                command_y_range=tuple(float(v) for v in args.command_y_range),
                command_yaw_range=tuple(float(v) for v in args.command_yaw_range),
                command_zero_probability=float(args.command_zero_probability),
                command_curriculum_start=command_curriculum_start,
                command_curriculum_samples=command_curriculum_samples,
                use_target_curriculum=not bool(args.no_target_curriculum),
                target_distance_start=float(args.target_distance_start),
                target_distance_end=float(args.target_distance_end),
                target_curriculum_samples=int(args.target_curriculum_samples),
                randomize_target_positions=bool(args.randomize_target_positions),
                target_angle_min=float(args.target_angle_min),
                target_angle_max=float(args.target_angle_max),
                reset_recurrent_state_on_rollout_start=bool(args.reset_recurrent_state_on_rollout_start),
                squash_actions=bool(args.squash_actions),
                log_std_init=float(args.log_std_init),
                resume_checkpoint=resume_checkpoint,
                checkpoint_path=checkpoint_template,
                checkpoint_interval=0,
                readback_diagnostics=bool(args.readback_diagnostics),
                execution_mode=str(args.execution_mode),
            )
        )
        train_seconds += time.perf_counter() - chunk_t0
        completed_iterations = int(result.trainer.iteration)
        checkpoint_path = _format_checkpoint_template(checkpoint_template, completed_iterations)
        resume_checkpoint = str(checkpoint_path)
        train_history.extend(asdict(item) for item in result.history)

        if (
            angular_fine_tune_iteration > 0
            and completed_iterations == angular_fine_tune_iteration
            and completed_iterations < int(args.max_iterations)
        ):
            continue

        gate_t0 = time.perf_counter()
        gate_result = _evaluate_checkpoint(checkpoint_path, args, device=device)
        gate_seconds += time.perf_counter() - gate_t0
        samples = _samples_for_iteration(args, completed_iterations)
        gate_entry = {
            "iteration": int(completed_iterations),
            "samples": int(samples),
            "checkpoint": str(checkpoint_path),
            "stats": asdict(gate_result.stats),
        }
        gate_history.append(gate_entry)
        if gate_result.stats.pass_gate and first_pass is None:
            first_pass = gate_entry
            if not bool(args.keep_going_after_pass):
                break

    total_seconds = max(time.perf_counter() - total_t0, 1.0e-12)
    trained_samples = _samples_for_iteration(args, completed_iterations)
    new_trained_samples = max(0, int(trained_samples) - int(start_samples))
    train_sps = float(new_trained_samples) / max(train_seconds, 1.0e-12)
    total_sps = float(new_trained_samples) / total_seconds
    estimated_target_train_seconds = (
        None if bool(args.evaluate_only) else float(args.target_samples) / max(train_sps, 1.0e-12)
    )
    estimated_target_total_seconds = (
        None if bool(args.evaluate_only) else float(args.target_samples) / max(total_sps, 1.0e-12)
    )
    pass_gate = first_pass is not None

    result = {
        "engine": "phoenx_g1_warp_ppo_train_to_gate",
        "metric": "train, save, reload, and evaluate G1 quality gate",
        "device": device.name,
        "execution_mode": str(args.execution_mode),
        "readback_diagnostics": bool(args.readback_diagnostics),
        "evaluate_only": bool(args.evaluate_only),
        "world_count": int(args.world_count),
        "rollout_steps": int(args.rollout_steps),
        "sim_substeps": int(args.sim_substeps),
        "physics_timestep": float(g1_recipe.FRAME_DT) / float(args.sim_substeps),
        "solver_iterations": int(args.solver_iterations),
        "velocity_iterations": int(args.velocity_iterations),
        "squash_actions": bool(args.squash_actions),
        "log_std_init": float(args.log_std_init),
        "action_scale": float(args.action_scale),
        "reward_mode": str(args.reward_mode),
        "w_alive": float(args.w_alive),
        "w_track_lin": float(args.w_track_lin),
        "w_track_ang": float(args.w_track_ang),
        "w_command_progress": float(args.w_command_progress),
        "w_lin_vel_z": float(args.w_lin_vel_z),
        "w_ang_vel_xy": float(args.w_ang_vel_xy),
        "angular_fine_tune_start_samples": int(angular_fine_tune_start_samples),
        "angular_fine_tune_iteration": int(angular_fine_tune_iteration),
        "angular_fine_tune_w_ang_vel_xy": float(args.angular_fine_tune_w_ang_vel_xy),
        "angular_fine_tune_lr_anneal_timesteps": int(args.angular_fine_tune_lr_anneal_timesteps),
        "w_orientation": float(args.w_orientation),
        "w_torque": float(args.w_torque),
        "w_action_rate": float(args.w_action_rate),
        "w_sparse_command_success": float(args.w_sparse_command_success),
        "w_target_progress": float(args.w_target_progress),
        "sparse_command_velocity_tolerance": float(args.sparse_command_velocity_tolerance),
        "sparse_command_yaw_tolerance": float(args.sparse_command_yaw_tolerance),
        "target_x": float(args.target_x),
        "target_y": float(args.target_y),
        "sparse_target_radius": float(args.sparse_target_radius),
        "sparse_target_success_upright_cos": float(args.sparse_target_success_upright_cos),
        "sparse_target_success_min_base_height": float(args.sparse_target_success_min_base_height),
        "sparse_target_success_max_base_height": float(args.sparse_target_success_max_base_height),
        "use_target_curriculum": not bool(args.no_target_curriculum),
        "target_distance_start": float(args.target_distance_start),
        "target_distance_end": float(args.target_distance_end),
        "target_curriculum_samples": int(args.target_curriculum_samples),
        "randomize_target_positions": bool(args.randomize_target_positions),
        "target_angle_min": float(args.target_angle_min),
        "target_angle_max": float(args.target_angle_max),
        "w_mechanical_power": float(args.w_mechanical_power),
        "w_gait_contact": float(args.w_gait_contact),
        "w_gait_swing": float(args.w_gait_swing),
        "w_gait_swing_contact": float(args.w_gait_swing_contact),
        "w_gait_hip": float(args.w_gait_hip),
        "w_base_height": float(args.w_base_height),
        "w_feet_air_time": float(args.w_feet_air_time),
        "feet_air_time_threshold": float(args.feet_air_time_threshold),
        "w_feet_slide": float(args.w_feet_slide),
        "w_joint_deviation_hip": float(args.w_joint_deviation_hip),
        "w_joint_deviation_waist": float(args.w_joint_deviation_waist),
        "w_joint_deviation_upper": float(args.w_joint_deviation_upper),
        "w_joint_acc_legs": float(args.w_joint_acc_legs),
        "w_joint_pos_limit_ankle": float(args.w_joint_pos_limit_ankle),
        "ground_friction": float(args.ground_friction),
        "foot_box_xy_scale": float(args.foot_box_xy_scale),
        "articulation_mode": str(getattr(args, "articulation_mode", "reduced")),
        "command_curriculum_start": float(command_curriculum_start),
        "command_curriculum_samples": int(command_curriculum_samples),
        "command_zero_probability": float(args.command_zero_probability),
        "reset_recurrent_state_on_rollout_start": bool(args.reset_recurrent_state_on_rollout_start),
        "value_loss_coeff": float(args.value_loss_coeff),
        "value_clip_range": float(args.value_clip_range),
        "actor_lr": float(args.actor_lr),
        "critic_lr": float(args.critic_lr),
        "anneal_lr": bool(args.anneal_lr),
        "lr_anneal_timesteps": int(args.lr_anneal_timesteps),
        "min_lr_ratio": float(args.min_lr_ratio),
        "optimizer": str(args.optimizer),
        "optimizer_eps": float(args.optimizer_eps),
        "optimizer_weight_decay": float(args.optimizer_weight_decay),
        "muon_momentum": float(args.muon_momentum),
        "max_iterations": int(args.max_iterations),
        "chunk_iterations": int(args.chunk_iterations),
        "start_iterations": int(start_iterations),
        "completed_iterations": int(completed_iterations),
        "start_samples": int(start_samples),
        "trained_samples": int(trained_samples),
        "new_trained_samples": int(new_trained_samples),
        "target_samples": int(args.target_samples),
        "train_seconds": float(train_seconds),
        "gate_seconds": float(gate_seconds),
        "total_wall_seconds": float(total_seconds),
        "train_env_samples_per_s": float(train_sps),
        "total_env_samples_per_s": float(total_sps),
        "estimated_target_train_seconds": estimated_target_train_seconds,
        "estimated_target_total_seconds": estimated_target_total_seconds,
        "checkpoint_path": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "pass_gate": bool(pass_gate),
        "first_pass": first_pass,
        "gate_history": gate_history,
        "train_history": train_history if bool(args.include_train_history) else [],
        "nanog1_reference_source": _NANOG1_REFERENCE,
        "nanog1_walk_samples": _NANOG1_WALK_SAMPLES,
        "nanog1_walk_seconds": _NANOG1_WALK_SECONDS,
        "nanog1_walk_env_samples_per_s": _NANOG1_WALK_SPS,
        "phoenx_estimated_target_slowdown_vs_nanog1": (
            None if estimated_target_train_seconds is None else estimated_target_train_seconds / _NANOG1_WALK_SECONDS
        ),
        "phoenx_train_sps_over_nanog1": train_sps / _NANOG1_WALK_SPS,
    }
    if args.fail_on_miss and not pass_gate:
        result["error"] = "quality gate was not reached within max_iterations"
    return result


def _default_max_iterations(world_count: int, rollout_steps: int, target_samples: int) -> int:
    samples_per_iteration = int(world_count) * int(rollout_steps)
    return max(1, int(math.ceil(float(target_samples) / float(samples_per_iteration))))


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=g1_recipe.WORLD_COUNT)
    parser.add_argument("--rollout-steps", type=int, default=g1_recipe.ROLLOUT_STEPS)
    parser.add_argument("--target-samples", type=int, default=_NANOG1_WALK_SAMPLES)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--chunk-iterations", type=int, default=25)
    parser.add_argument("--hidden-layers", type=_parse_hidden_layers, default=g1_recipe.HIDDEN_LAYERS)
    parser.add_argument("--activation", choices=("relu", "elu"), default=g1_recipe.ACTIVATION)
    parser.add_argument("--policy-network", choices=("mlp", "puffer_mingru"), default=g1_recipe.POLICY_NETWORK)
    parser.add_argument("--train-epochs", type=int, default=g1_recipe.TRAIN_EPOCHS)
    parser.add_argument("--actor-lr", type=float, default=g1_recipe.ACTOR_LR)
    parser.add_argument("--critic-lr", type=float, default=g1_recipe.CRITIC_LR)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=g1_recipe.ANNEAL_LR)
    parser.add_argument("--lr-anneal-timesteps", type=int, default=g1_recipe.LR_ANNEAL_TIMESTEPS)
    parser.add_argument("--min-lr-ratio", type=float, default=g1_recipe.MIN_LR_RATIO)
    parser.add_argument("--mirror-loss-coeff", type=float, default=g1_recipe.MIRROR_LOSS_COEFF)
    parser.add_argument("--minibatch-size", type=int, default=g1_recipe.MINIBATCH_SIZE)
    parser.add_argument("--replay-ratio", type=float, default=g1_recipe.REPLAY_RATIO)
    parser.add_argument("--priority-alpha", type=float, default=g1_recipe.PRIORITY_ALPHA)
    parser.add_argument("--priority-beta", type=float, default=g1_recipe.PRIORITY_BETA)
    parser.add_argument("--no-manual-actor-backward", action="store_true")
    parser.add_argument("--no-manual-critic-backward", action="store_true")
    parser.add_argument(
        "--manual-mlp-weight-grad-dtype",
        choices=("float32", "bfloat16"),
        default=g1_recipe.MANUAL_MLP_WEIGHT_GRAD_DTYPE,
    )
    parser.add_argument(
        "--manual-mlp-forward-dtype",
        choices=("float32", "bfloat16"),
        default=g1_recipe.MANUAL_MLP_FORWARD_DTYPE,
    )
    parser.add_argument("--vtrace-rho-clip", type=float, default=g1_recipe.VTRACE_RHO_CLIP)
    parser.add_argument("--vtrace-c-clip", type=float, default=g1_recipe.VTRACE_C_CLIP)
    parser.add_argument("--reward-clip", type=float, default=g1_recipe.REWARD_CLIP)
    parser.add_argument("--max-grad-norm", type=float, default=g1_recipe.MAX_GRAD_NORM)
    parser.add_argument("--value-loss-coeff", type=float, default=g1_recipe.VALUE_LOSS_COEFF)
    parser.add_argument("--value-clip-range", type=float, default=g1_recipe.VALUE_CLIP_RANGE)
    parser.add_argument("--optimizer", choices=("adam", "muon"), default=g1_recipe.OPTIMIZER)
    parser.add_argument("--optimizer-eps", type=float, default=g1_recipe.OPTIMIZER_EPS)
    parser.add_argument("--optimizer-weight-decay", type=float, default=g1_recipe.OPTIMIZER_WEIGHT_DECAY)
    parser.add_argument("--muon-momentum", type=float, default=g1_recipe.MUON_MOMENTUM)
    parser.add_argument(
        "--squash-actions",
        action=argparse.BooleanOptionalAction,
        default=g1_recipe.SQUASH_ACTIONS,
        help="Use tanh-squashed PPO actions instead of the nanoG1-compatible raw Gaussian policy.",
    )
    parser.add_argument("--log-std-init", type=float, default=g1_recipe.LOG_STD_INIT)
    parser.add_argument(
        "--reset-recurrent-state-on-rollout-start",
        action=argparse.BooleanOptionalAction,
        default=g1_recipe.RESET_RECURRENT_STATE_ON_ROLLOUT_START,
        help="Clear recurrent policy state at each PPO rollout boundary.",
    )
    parser.add_argument("--command-x", type=float, default=g1_recipe.COMMAND[0])
    parser.add_argument("--command-y", type=float, default=g1_recipe.COMMAND[1])
    parser.add_argument("--command-yaw", type=float, default=g1_recipe.COMMAND[2])
    parser.add_argument("--command-x-range", type=float, nargs=2, default=g1_recipe.COMMAND_X_RANGE)
    parser.add_argument("--command-y-range", type=float, nargs=2, default=g1_recipe.COMMAND_Y_RANGE)
    parser.add_argument("--command-yaw-range", type=float, nargs=2, default=g1_recipe.COMMAND_YAW_RANGE)
    parser.add_argument(
        "--command-curriculum-start",
        type=float,
        default=g1_recipe.COMMAND_CURRICULUM_START,
        help="Initial nanoG1 command-range scale for randomized G1 commands.",
    )
    parser.add_argument(
        "--command-curriculum-samples",
        type=int,
        default=g1_recipe.COMMAND_CURRICULUM_SAMPLES,
        help="Samples used to ramp randomized G1 commands to full range; 0 disables the ramp.",
    )
    parser.add_argument("--command-zero-probability", type=float, default=g1_recipe.COMMAND_ZERO_PROBABILITY)
    parser.add_argument("--no-command-randomization", action="store_true")
    parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    parser.add_argument(
        "--articulation-mode",
        choices=("maximal", "hybrid", "reduced"),
        default="reduced",
        help="PhoenX articulation mode used by both training and the frozen quality gate.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=("nanog1_dense", "sparse_command", "sparse_target", "dense_sparse_command"),
        default=g1_recipe.REWARD_MODE,
    )
    parser.add_argument("--w-alive", type=float, default=g1_recipe.W_ALIVE)
    parser.add_argument("--w-track-lin", type=float, default=g1_recipe.W_TRACK_LIN)
    parser.add_argument("--w-track-ang", type=float, default=g1_recipe.W_TRACK_ANG)
    parser.add_argument("--w-command-progress", type=float, default=g1_recipe.W_COMMAND_PROGRESS)
    parser.add_argument("--w-lin-vel-z", type=float, default=g1_recipe.W_LIN_VEL_Z)
    parser.add_argument("--w-ang-vel-xy", type=float, default=g1_recipe.W_ANG_VEL_XY)
    parser.add_argument(
        "--angular-fine-tune-start-samples",
        type=int,
        default=g1_recipe.ANGULAR_FINE_TUNE_START_SAMPLES,
        help="Switch to the final roll/pitch regularization phase after this many samples; 0 disables it.",
    )
    parser.add_argument(
        "--angular-fine-tune-w-ang-vel-xy",
        type=float,
        default=g1_recipe.ANGULAR_FINE_TUNE_W_ANG_VEL_XY,
    )
    parser.add_argument(
        "--angular-fine-tune-lr-anneal-timesteps",
        type=int,
        default=g1_recipe.ANGULAR_FINE_TUNE_LR_ANNEAL_TIMESTEPS,
    )
    parser.add_argument("--w-orientation", type=float, default=g1_recipe.W_ORIENTATION)
    parser.add_argument("--w-torque", type=float, default=g1_recipe.W_TORQUE)
    parser.add_argument("--w-action-rate", type=float, default=g1_recipe.W_ACTION_RATE)
    parser.add_argument("--w-sparse-command-success", type=float, default=g1_recipe.W_SPARSE_COMMAND_SUCCESS)
    parser.add_argument("--w-target-progress", type=float, default=g1_recipe.W_TARGET_PROGRESS)
    parser.add_argument(
        "--sparse-command-velocity-tolerance",
        type=float,
        default=g1_recipe.SPARSE_COMMAND_VELOCITY_TOLERANCE,
    )
    parser.add_argument(
        "--sparse-command-yaw-tolerance",
        type=float,
        default=g1_recipe.SPARSE_COMMAND_YAW_TOLERANCE,
    )
    parser.add_argument("--target-x", type=float, default=g1_recipe.SPARSE_TARGET_POSITION[0])
    parser.add_argument("--target-y", type=float, default=g1_recipe.SPARSE_TARGET_POSITION[1])
    parser.add_argument("--sparse-target-radius", type=float, default=g1_recipe.SPARSE_TARGET_RADIUS)
    parser.add_argument(
        "--sparse-target-success-upright-cos", type=float, default=g1_recipe.SPARSE_TARGET_SUCCESS_UPRIGHT_COS
    )
    parser.add_argument(
        "--sparse-target-success-min-base-height",
        type=float,
        default=g1_recipe.SPARSE_TARGET_SUCCESS_MIN_BASE_HEIGHT,
    )
    parser.add_argument(
        "--sparse-target-success-max-base-height",
        type=float,
        default=g1_recipe.SPARSE_TARGET_SUCCESS_MAX_BASE_HEIGHT,
    )
    parser.add_argument("--target-distance-start", type=float, default=g1_recipe.SPARSE_TARGET_CURRICULUM_START)
    parser.add_argument("--target-distance-end", type=float, default=g1_recipe.SPARSE_TARGET_CURRICULUM_END)
    parser.add_argument("--target-curriculum-samples", type=int, default=g1_recipe.SPARSE_TARGET_CURRICULUM_SAMPLES)
    parser.add_argument("--no-target-curriculum", action="store_true")
    parser.add_argument(
        "--randomize-target-positions",
        action=argparse.BooleanOptionalAction,
        default=g1_recipe.SPARSE_TARGET_RANDOMIZE,
    )
    parser.add_argument("--target-angle-min", type=float, default=g1_recipe.SPARSE_TARGET_ANGLE_MIN)
    parser.add_argument("--target-angle-max", type=float, default=g1_recipe.SPARSE_TARGET_ANGLE_MAX)
    parser.add_argument("--w-mechanical-power", type=float, default=g1_recipe.W_MECHANICAL_POWER)
    parser.add_argument("--w-gait-contact", type=float, default=g1_recipe.W_GAIT_CONTACT)
    parser.add_argument("--w-gait-swing", type=float, default=g1_recipe.W_GAIT_SWING)
    parser.add_argument("--w-gait-swing-contact", type=float, default=g1_recipe.W_GAIT_SWING_CONTACT)
    parser.add_argument("--w-gait-hip", type=float, default=g1_recipe.W_GAIT_HIP)
    parser.add_argument("--w-base-height", type=float, default=g1_recipe.W_BASE_HEIGHT)
    parser.add_argument("--w-feet-air-time", type=float, default=g1_recipe.W_FEET_AIR_TIME)
    parser.add_argument("--feet-air-time-threshold", type=float, default=g1_recipe.FEET_AIR_TIME_THRESHOLD)
    parser.add_argument("--w-feet-slide", type=float, default=g1_recipe.W_FEET_SLIDE)
    parser.add_argument("--w-joint-deviation-hip", type=float, default=g1_recipe.W_JOINT_DEVIATION_HIP)
    parser.add_argument("--w-joint-deviation-waist", type=float, default=g1_recipe.W_JOINT_DEVIATION_WAIST)
    parser.add_argument("--w-joint-deviation-upper", type=float, default=g1_recipe.W_JOINT_DEVIATION_UPPER)
    parser.add_argument("--w-joint-acc-legs", type=float, default=g1_recipe.W_JOINT_ACC_LEGS)
    parser.add_argument("--w-joint-pos-limit-ankle", type=float, default=g1_recipe.W_JOINT_POS_LIMIT_ANKLE)
    parser.add_argument(
        "--actuation-model",
        choices=("explicit_torque", "constraint_drive"),
        default=g1_recipe.ACTUATION_MODEL,
        help="G1 actuator path used for train and gate environments.",
    )
    parser.add_argument("--action-scale", type=float, default=g1_recipe.ACTION_SCALE)
    parser.add_argument("--controlled-action-count", type=int, default=g1_recipe.CONTROLLED_ACTION_COUNT)
    parser.add_argument("--observation-mode", choices=("nanog1", "isaaclab_flat"), default=g1_recipe.OBSERVATION_MODE)
    parser.add_argument("--parse-meshes", action="store_true")
    parser.add_argument("--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=g1_recipe.CONTACT_GEOMETRY)
    parser.add_argument("--ground-friction", type=float, default=g1_recipe.GROUND_FRICTION)
    parser.add_argument("--foot-box-xy-scale", type=float, default=g1_recipe.FOOT_BOX_XY_SCALE)
    parser.add_argument("--rigid-contact-max-per-world", type=int, default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
    parser.add_argument("--threads-per-world", type=_parse_int_or_auto, default=g1_recipe.THREADS_PER_WORLD)
    parser.add_argument("--multi-world-scheduler", default=g1_recipe.MULTI_WORLD_SCHEDULER)
    parser.add_argument("--prepare-refresh-stride", type=_parse_int_or_auto, default=g1_recipe.PREPARE_REFRESH_STRIDE)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "graph_leapfrog"),
        default="graph_leapfrog",
        help="Use eager PPO or the separate-graph rollout/update schedule.",
    )
    parser.add_argument("--readback-diagnostics", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=g1_recipe.SEED)
    parser.add_argument("--battery-steps", type=int, default=1000)
    parser.add_argument("--seeds-per-command", type=int, default=4)
    parser.add_argument("--diagnostic-steps", type=int, default=2000)
    parser.add_argument("--diagnostic-world-count", type=int, default=1)
    parser.add_argument("--stochastic-gate", action="store_true")
    parser.add_argument("--gate-seed", type=int, default=1000)
    parser.add_argument("--max-battery-falls", type=int, default=1)
    parser.add_argument("--min-battery-perf", type=float, default=0.90)
    parser.add_argument("--max-action-jerk-rms", type=float, default=0.21)
    parser.add_argument("--max-ang-vel-xy-rms", type=float, default=0.21)
    parser.add_argument("--max-yaw-rate-rms", type=float, default=0.20)
    parser.add_argument("--max-leg-qvel-rms", type=float, default=1.22)
    parser.add_argument("--keep-going-after-pass", action="store_true")
    parser.add_argument("--fail-on-miss", action="store_true")
    parser.add_argument("--include-train-history", action="store_true")
    parser.add_argument("--json-indent", type=int, default=2)
    return parser


def _parse_args() -> argparse.Namespace:
    args = _make_parser().parse_args()
    if args.max_iterations is None:
        args.max_iterations = _default_max_iterations(args.world_count, args.rollout_steps, args.target_samples)
    return args


def main() -> int:
    args = _parse_args()
    result = benchmark_train_to_gate(args)
    print(json.dumps(result, indent=args.json_indent, sort_keys=True))
    return 1 if args.fail_on_miss and not result["pass_gate"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
