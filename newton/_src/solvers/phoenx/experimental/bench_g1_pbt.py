# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sequential pure-Warp PBT probe for PhoenX G1 PPO.

This is an experimental controller inspired by Sample Factory's PBT notes, but
it does not import Sample Factory or PyTorch. Each population member is a normal
PhoenX PPO checkpoint plus a small set of mutable PPO/reward hyperparameters.
Members train for a fixed number of PPO iterations, are evaluated with the G1
quality gate and/or fixed-command no-reset progress metrics, and the weakest
members can be replaced by mutated copies of the strongest members.

The objective is deliberately task-level evaluation, not training reward. This
avoids reward-weight inflation and lets shaped rewards act as a search surface
rather than the final target.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe


@dataclass
class TrialConfigPBT:
    """Mutable knobs for one PBT member."""

    action_scale: float
    log_std_init: float
    actor_lr: float
    critic_lr: float
    entropy_coeff: float
    max_grad_norm: float
    reward_mode: str
    w_alive: float
    w_termination: float
    w_track_lin: float
    w_track_ang: float
    w_ang_vel_xy: float
    w_orientation: float
    w_action_rate: float
    w_sparse_command_success: float
    sparse_command_velocity_tolerance: float
    command_zero_probability: float
    command_curriculum_start: float


@dataclass
class TrialStatePBT:
    """Persistent state for one PBT member."""

    policy_id: int
    config: TrialConfigPBT
    checkpoint: str | None = None
    score: float = float("-inf")
    progress_score: float | None = None
    gate_score: float | None = None
    battery_perf: float = 0.0
    battery_falls: int = 0
    ang_vel_xy_rms: float = 0.0
    iteration: int = 0


def _base_trial_config(args: argparse.Namespace | None = None) -> TrialConfigPBT:
    cfg = TrialConfigPBT(
        action_scale=g1_recipe.ACTION_SCALE,
        log_std_init=g1_recipe.LOG_STD_INIT,
        actor_lr=g1_recipe.ACTOR_LR,
        critic_lr=g1_recipe.CRITIC_LR,
        entropy_coeff=g1_recipe.ENTROPY_COEFF,
        max_grad_norm=g1_recipe.MAX_GRAD_NORM,
        reward_mode=g1_recipe.REWARD_MODE,
        w_alive=g1_recipe.W_ALIVE,
        w_termination=g1_recipe.W_TERMINATION,
        w_track_lin=g1_recipe.W_TRACK_LIN,
        w_track_ang=g1_recipe.W_TRACK_ANG,
        w_ang_vel_xy=g1_recipe.W_ANG_VEL_XY,
        w_orientation=g1_recipe.W_ORIENTATION,
        w_action_rate=g1_recipe.W_ACTION_RATE,
        w_sparse_command_success=g1_recipe.W_SPARSE_COMMAND_SUCCESS,
        sparse_command_velocity_tolerance=g1_recipe.SPARSE_COMMAND_VELOCITY_TOLERANCE,
        command_zero_probability=g1_recipe.COMMAND_ZERO_PROBABILITY,
        command_curriculum_start=g1_recipe.COMMAND_CURRICULUM_START,
    )
    if args is None:
        return cfg
    return replace(
        cfg,
        action_scale=float(args.base_action_scale),
        log_std_init=float(args.base_log_std_init),
        reward_mode=str(args.base_reward_mode),
    )


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


def _perturb_float(
    value: float, *, lo: float, hi: float, min_scale: float, max_scale: float, rng: random.Random
) -> float:
    scale = rng.uniform(float(min_scale), float(max_scale))
    value = float(value) / scale if rng.random() < 0.5 else float(value) * scale
    return _clip(value, lo, hi)


def _maybe_perturb(
    value: float,
    *,
    default: float,
    lo: float,
    hi: float,
    mutation_rate: float,
    min_scale: float,
    max_scale: float,
    rng: random.Random,
) -> float:
    if rng.random() > float(mutation_rate):
        return float(value)
    if value != default and rng.random() < 0.01:
        return float(default)
    return _perturb_float(value, lo=lo, hi=hi, min_scale=min_scale, max_scale=max_scale, rng=rng)


def _mutate_config(cfg: TrialConfigPBT, args: argparse.Namespace, rng: random.Random) -> TrialConfigPBT:
    base = _base_trial_config(args)
    values = asdict(cfg)
    specs = {
        "action_scale": (0.12, 0.35),
        "actor_lr": (1.0e-4, 5.0e-2),
        "critic_lr": (1.0e-4, 5.0e-2),
        "entropy_coeff": (0.0, 5.0e-3),
        "max_grad_norm": (0.05, 2.0),
        "w_alive": (0.0, 5.0),
        "w_termination": (-5.0, -0.1),
        "w_track_lin": (1.0, 10.0),
        "w_track_ang": (0.0, 4.0),
        "w_ang_vel_xy": (-5.0, -0.05),
        "w_orientation": (-15.0, -0.5),
        "w_action_rate": (-0.1, -0.001),
        "w_sparse_command_success": (1.0, 12.0),
        "sparse_command_velocity_tolerance": (0.15, 0.6),
        "command_zero_probability": (0.0, 0.25),
        "command_curriculum_start": (0.1, 0.8),
    }
    for key, (lo, hi) in specs.items():
        values[key] = _maybe_perturb(
            float(values[key]),
            default=float(getattr(base, key)),
            lo=lo,
            hi=hi,
            mutation_rate=float(args.mutation_rate),
            min_scale=float(args.perturb_min),
            max_scale=float(args.perturb_max),
            rng=rng,
        )
    if rng.random() <= float(args.mutation_rate):
        values["log_std_init"] = _clip(float(values["log_std_init"]) + rng.uniform(-0.25, 0.25), -1.2, 0.3)
    if rng.random() <= float(args.reward_mode_mutation_rate):
        modes = tuple(str(mode) for mode in args.reward_mode_choices)
        values["reward_mode"] = rng.choice(modes)
    return TrialConfigPBT(**values)


def _initial_trial_config(policy_id: int, args: argparse.Namespace, rng: random.Random) -> TrialConfigPBT:
    base = _base_trial_config(args)
    if not bool(args.seed_presets):
        return base if policy_id == 0 else _mutate_config(base, args, rng)
    presets = (
        base,
        replace(
            base,
            w_alive=0.0,
            w_track_lin=6.0,
            command_zero_probability=0.0,
            log_std_init=-0.3,
        ),
        replace(
            base,
            action_scale=0.18,
            log_std_init=-0.5,
        ),
        replace(
            base,
            reward_mode="sparse_command",
            w_alive=0.0,
            w_sparse_command_success=5.0,
            sparse_command_velocity_tolerance=0.35,
            command_zero_probability=0.0,
            log_std_init=-0.5,
        ),
    )
    if policy_id < len(presets):
        return presets[policy_id]
    return _mutate_config(base, args, rng)


def _trial_env_config(args: argparse.Namespace, cfg: TrialConfigPBT) -> rl.ConfigEnvG1PhoenX:
    return g1_recipe.default_g1_env_config(
        world_count=int(args.world_count),
        sim_substeps=int(args.sim_substeps),
        solver_iterations=int(args.solver_iterations),
        velocity_iterations=int(args.velocity_iterations),
        action_scale=float(cfg.action_scale),
        reward_mode=str(cfg.reward_mode),
        w_alive=float(cfg.w_alive),
        w_termination=float(cfg.w_termination),
        w_track_lin=float(cfg.w_track_lin),
        w_track_ang=float(cfg.w_track_ang),
        w_ang_vel_xy=float(cfg.w_ang_vel_xy),
        w_orientation=float(cfg.w_orientation),
        w_action_rate=float(cfg.w_action_rate),
        w_sparse_command_success=float(cfg.w_sparse_command_success),
        sparse_command_velocity_tolerance=float(cfg.sparse_command_velocity_tolerance),
    )


def _trial_ppo_config(cfg: TrialConfigPBT) -> rl.ConfigPPO:
    return g1_recipe.default_g1_ppo_config(
        actor_lr=float(cfg.actor_lr),
        critic_lr=float(cfg.critic_lr),
        entropy_coeff=float(cfg.entropy_coeff),
        max_grad_norm=float(cfg.max_grad_norm),
    )


def _gate_score(stats: rl.StatsEvaluateG1GatePPO, args: argparse.Namespace) -> float:
    fall_rate = float(stats.battery_falls) / float(max(int(stats.battery_samples), 1))
    return (
        float(stats.battery_perf)
        - float(args.fall_penalty) * fall_rate
        - float(args.wobble_penalty) * float(stats.ang_vel_xy_rms)
    )


def _progress_commands(args: argparse.Namespace) -> tuple[tuple[float, float, float], ...]:
    commands = args.progress_command
    if commands is None:
        commands = ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.5, 0.0, 0.0), (0.8, 0.0, 0.0))
    return tuple((float(cmd[0]), float(cmd[1]), float(cmd[2])) for cmd in commands)


def _fixed_command_score(
    stats: rl.StatsEvaluateG1PPO, command: tuple[float, float, float], frame_dt: float, args: argparse.Namespace
) -> float:
    target_speed = math.hypot(float(command[0]), float(command[1]))
    fall = float(stats.fall_fraction)
    survival = float(stats.mean_survival_steps) / float(max(int(stats.steps), 1))
    if target_speed <= 1.0e-6:
        return (
            float(args.stand_weight) * (float(stats.mean_tracking_perf) + survival)
            - float(args.progress_fall_penalty) * fall
            - float(args.stand_path_penalty) * float(stats.mean_path_length)
        )

    eval_seconds = max(float(stats.steps) * float(frame_dt), 1.0e-12)
    target_distance = max(target_speed * eval_seconds, 1.0e-6)
    velocity_fraction = float(stats.mean_command_aligned_velocity) / target_speed
    progress_fraction = float(stats.mean_command_aligned_displacement) / target_distance
    velocity_score = 1.0 - min(abs(velocity_fraction - 1.0), 2.0)
    progress_score = _clip(progress_fraction, -1.0, 1.25)
    lateral_fraction = float(stats.mean_lateral_displacement_abs) / max(target_distance, 0.1)
    return (
        0.5 * velocity_score
        + 0.5 * progress_score
        + 0.25 * survival
        - float(args.progress_fall_penalty) * fall
        - float(args.lateral_penalty) * lateral_fraction
    )


def _evaluate_progress_score(
    trainer: rl.TrainerPPO,
    env_config: rl.ConfigEnvG1PhoenX,
    args: argparse.Namespace,
    device: wp.context.Device,
) -> tuple[float, tuple[dict[str, Any], ...]]:
    commands = _progress_commands(args)
    total = 0.0
    total_weight = 0.0
    evaluations: list[dict[str, Any]] = []
    for command in commands:
        eval_config = replace(
            env_config,
            world_count=int(args.progress_world_count),
            command=command,
            auto_reset=False,
            max_episode_steps=0,
        )
        result = rl.evaluate_g1_ppo(
            trainer,
            rl.ConfigEvaluateG1PPO(
                env_config=eval_config,
                steps=int(args.progress_steps),
                device=device,
                deterministic=not bool(args.stochastic_progress),
                seed=int(args.progress_seed),
            ),
        )
        stats = result.stats
        score = _fixed_command_score(stats, command, float(env_config.frame_dt), args)
        weight = float(args.stand_weight) if math.hypot(float(command[0]), float(command[1])) <= 1.0e-6 else 1.0
        total += weight * score
        total_weight += weight
        entry = asdict(stats)
        entry["command"] = command
        entry["score"] = float(score)
        evaluations.append(entry)
    return total / max(total_weight, 1.0e-12), tuple(evaluations)


def _combine_scores(gate_score: float | None, progress_score: float | None, args: argparse.Namespace) -> float:
    mode = str(args.score_mode)
    if mode == "gate":
        if gate_score is None:
            raise RuntimeError("gate score was not evaluated")
        return float(gate_score)
    if mode == "progress":
        if progress_score is None:
            raise RuntimeError("progress score was not evaluated")
        return float(progress_score)
    if gate_score is None or progress_score is None:
        raise RuntimeError("combined score requires both gate and progress evaluation")
    return float(args.gate_score_weight) * float(gate_score) + float(args.progress_score_weight) * float(progress_score)


def _train_and_score_member(
    member: TrialStatePBT, args: argparse.Namespace, epoch: int, device: wp.context.Device
) -> dict[str, Any]:
    work_dir = Path(args.work_dir)
    policy_dir = work_dir / f"policy_{member.policy_id:02d}"
    checkpoint_template = str(policy_dir / "checkpoint_{iteration}.npz")
    env_config = _trial_env_config(args, member.config)
    result = rl.train_g1_ppo(
        rl.ConfigTrainG1PPO(
            iterations=int(args.epoch_iterations),
            rollout_steps=int(args.rollout_steps),
            hidden_layers=tuple(int(v) for v in args.hidden_layers),
            log_std_init=float(member.config.log_std_init),
            env_config=env_config,
            ppo_config=_trial_ppo_config(member.config),
            device=device,
            seed=int(args.seed) + member.policy_id * 1009 + epoch * 9173,
            log_interval=0,
            randomize_commands=not bool(args.no_command_randomization),
            command_x_range=tuple(float(v) for v in args.command_x_range),
            command_y_range=tuple(float(v) for v in args.command_y_range),
            command_yaw_range=tuple(float(v) for v in args.command_yaw_range),
            command_zero_probability=float(member.config.command_zero_probability),
            command_curriculum_start=float(member.config.command_curriculum_start),
            command_curriculum_samples=int(args.command_curriculum_samples),
            squash_actions=bool(args.squash_actions),
            resume_checkpoint=member.checkpoint,
            checkpoint_path=checkpoint_template,
            checkpoint_interval=0,
            readback_diagnostics=False,
            execution_mode=str(args.execution_mode),
        )
    )
    iteration = int(result.trainer.iteration)
    checkpoint_path = policy_dir / f"checkpoint_{iteration}.npz"
    latest_path = policy_dir / "latest.npz"
    shutil.copyfile(checkpoint_path, latest_path)

    gate_payload: dict[str, Any] | None = None
    gate_score: float | None = None
    if str(args.score_mode) != "progress":
        gate = rl.evaluate_g1_gate_ppo(
            result.trainer,
            rl.ConfigEvaluateG1GatePPO(
                env_config=env_config,
                battery_steps=int(args.battery_steps),
                seeds_per_command=int(args.seeds_per_command),
                diagnostic_steps=int(args.diagnostic_steps),
                diagnostic_world_count=int(args.diagnostic_world_count),
                device=device,
                deterministic=not bool(args.stochastic_gate),
                seed=int(args.gate_seed),
                max_battery_falls=int(args.max_battery_falls),
                min_battery_perf=float(args.min_battery_perf),
                max_ang_vel_xy_rms=float(args.max_ang_vel_xy_rms),
            ),
        )
        stats = gate.stats
        gate_payload = asdict(stats)
        gate_score = _gate_score(stats, args)
        member.battery_perf = float(stats.battery_perf)
        member.battery_falls = int(stats.battery_falls)
        member.ang_vel_xy_rms = float(stats.ang_vel_xy_rms)

    progress_payload: tuple[dict[str, Any], ...] = ()
    progress_score: float | None = None
    if str(args.score_mode) != "gate":
        progress_score, progress_payload = _evaluate_progress_score(result.trainer, env_config, args, device)

    member.checkpoint = str(latest_path)
    member.iteration = iteration
    member.gate_score = gate_score
    member.progress_score = progress_score
    member.score = _combine_scores(gate_score, progress_score, args)
    return {
        "policy_id": int(member.policy_id),
        "iteration": int(iteration),
        "checkpoint": str(latest_path),
        "score": float(member.score),
        "gate_score": gate_score,
        "progress_score": progress_score,
        "config": asdict(member.config),
        "gate": gate_payload,
        "progress": progress_payload,
    }


def _evolve(population: list[TrialStatePBT], args: argparse.Namespace, rng: random.Random) -> list[dict[str, Any]]:
    if len(population) < 2:
        return []
    ranked = sorted(population, key=lambda member: member.score, reverse=True)
    replace_count = max(1, math.ceil(float(args.replace_fraction) * len(population)))
    best = ranked[:replace_count]
    worst = ranked[-replace_count:]
    events: list[dict[str, Any]] = []
    for member in worst:
        if member in best or member.policy_id == 0:
            continue
        donor = rng.choice(best)
        if donor.checkpoint is None:
            continue
        reward_gap = float(donor.score) - float(member.score)
        if reward_gap < float(args.min_replace_score_gap):
            continue
        member_dir = Path(args.work_dir) / f"policy_{member.policy_id:02d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        copied_checkpoint = member_dir / "latest.npz"
        shutil.copyfile(donor.checkpoint, copied_checkpoint)
        member.checkpoint = str(copied_checkpoint)
        member.iteration = int(donor.iteration)
        member.config = _mutate_config(donor.config, args, rng)
        events.append(
            {
                "policy_id": int(member.policy_id),
                "donor_policy_id": int(donor.policy_id),
                "score_gap": reward_gap,
                "config": asdict(member.config),
            }
        )
    middle = ranked[replace_count : len(ranked) - replace_count]
    for member in middle:
        if member.policy_id != 0:
            member.config = _mutate_config(member.config, args, rng)
    return events


def run_pbt(args: argparse.Namespace) -> dict[str, Any]:
    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("G1 PBT probe requires CUDA with Warp mempool enabled")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    population = []
    for policy_id in range(int(args.population_size)):
        cfg = _initial_trial_config(policy_id, args, rng)
        population.append(TrialStatePBT(policy_id=policy_id, config=cfg))

    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    epochs = int(args.epochs)
    for epoch in range(epochs):
        members = [_train_and_score_member(member, args, epoch, device) for member in population]
        evolve_events = _evolve(population, args, rng) if epoch + 1 < epochs else []
        history.append({"epoch": epoch, "members": members, "evolve": evolve_events})
        state = {
            "epoch": epoch,
            "population": [asdict(member) for member in population],
            "history": history,
        }
        (work_dir / "pbt_state.json").write_text(json.dumps(state, indent=2))

    best = max(population, key=lambda member: member.score)
    return {
        "engine": "phoenx_g1_warp_ppo_pbt",
        "device": device.name,
        "work_dir": str(work_dir),
        "population_size": int(args.population_size),
        "epochs": int(args.epochs),
        "epoch_iterations": int(args.epoch_iterations),
        "elapsed_seconds": time.perf_counter() - t0,
        "best_policy_id": int(best.policy_id),
        "best_score": float(best.score),
        "best_checkpoint": best.checkpoint,
        "best_config": asdict(best.config),
        "population": [asdict(member) for member in population],
        "history": history,
    }


def _parse_hidden_layers(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in str(text).split(",") if item)
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("hidden layers must be a comma-separated list of positive widths")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="/tmp/phoenx_g1_pbt")
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--epoch-iterations", type=int, default=40)
    parser.add_argument("--world-count", type=int, default=2048)
    parser.add_argument("--rollout-steps", type=int, default=g1_recipe.ROLLOUT_STEPS)
    parser.add_argument("--hidden-layers", type=_parse_hidden_layers, default=g1_recipe.HIDDEN_LAYERS)
    parser.add_argument("--base-action-scale", type=float, default=g1_recipe.ACTION_SCALE)
    parser.add_argument("--base-log-std-init", type=float, default=g1_recipe.LOG_STD_INIT)
    parser.add_argument(
        "--base-reward-mode",
        choices=("nanog1_dense", "sparse_command", "sparse_target"),
        default=g1_recipe.REWARD_MODE,
    )
    parser.add_argument(
        "--reward-mode-choices",
        choices=("nanog1_dense", "sparse_command", "sparse_target"),
        nargs="+",
        default=("nanog1_dense", "sparse_command"),
    )
    parser.add_argument("--seed-presets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sim-substeps", type=int, default=g1_recipe.SIM_SUBSTEPS)
    parser.add_argument("--solver-iterations", type=int, default=g1_recipe.SOLVER_ITERATIONS)
    parser.add_argument("--velocity-iterations", type=int, default=g1_recipe.VELOCITY_ITERATIONS)
    parser.add_argument("--execution-mode", choices=("eager", "graph_leapfrog"), default="graph_leapfrog")
    parser.add_argument("--squash-actions", action=argparse.BooleanOptionalAction, default=g1_recipe.SQUASH_ACTIONS)
    parser.add_argument("--command-x-range", type=float, nargs=2, default=g1_recipe.COMMAND_X_RANGE)
    parser.add_argument("--command-y-range", type=float, nargs=2, default=g1_recipe.COMMAND_Y_RANGE)
    parser.add_argument("--command-yaw-range", type=float, nargs=2, default=g1_recipe.COMMAND_YAW_RANGE)
    parser.add_argument("--command-curriculum-samples", type=int, default=g1_recipe.COMMAND_CURRICULUM_SAMPLES)
    parser.add_argument("--no-command-randomization", action="store_true")
    parser.add_argument("--battery-steps", type=int, default=400)
    parser.add_argument("--seeds-per-command", type=int, default=2)
    parser.add_argument("--diagnostic-steps", type=int, default=500)
    parser.add_argument("--diagnostic-world-count", type=int, default=1)
    parser.add_argument("--stochastic-gate", action="store_true")
    parser.add_argument("--gate-seed", type=int, default=1000)
    parser.add_argument("--max-battery-falls", type=int, default=1)
    parser.add_argument("--min-battery-perf", type=float, default=0.90)
    parser.add_argument("--max-ang-vel-xy-rms", type=float, default=0.21)
    parser.add_argument("--score-mode", choices=("gate", "progress", "gate_progress"), default="gate_progress")
    parser.add_argument("--gate-score-weight", type=float, default=1.0)
    parser.add_argument("--progress-score-weight", type=float, default=1.0)
    parser.add_argument(
        "--progress-command",
        action="append",
        nargs=3,
        type=float,
        metavar=("X", "Y", "YAW"),
        help="Append a fixed no-reset command used by the progress objective.",
    )
    parser.add_argument("--progress-steps", type=int, default=300)
    parser.add_argument("--progress-world-count", type=int, default=64)
    parser.add_argument("--progress-seed", type=int, default=2000)
    parser.add_argument("--stochastic-progress", action="store_true")
    parser.add_argument("--fall-penalty", type=float, default=10.0)
    parser.add_argument("--wobble-penalty", type=float, default=0.25)
    parser.add_argument("--progress-fall-penalty", type=float, default=3.0)
    parser.add_argument("--lateral-penalty", type=float, default=0.2)
    parser.add_argument("--stand-weight", type=float, default=0.25)
    parser.add_argument("--stand-path-penalty", type=float, default=0.25)
    parser.add_argument("--replace-fraction", type=float, default=0.3)
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--reward-mode-mutation-rate", type=float, default=0.05)
    parser.add_argument("--perturb-min", type=float, default=1.05)
    parser.add_argument("--perturb-max", type=float, default=1.35)
    parser.add_argument("--min-replace-score-gap", type=float, default=0.02)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=g1_recipe.SEED)
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_pbt(args)
    print(json.dumps(result, indent=int(args.json_indent), sort_keys=True))


if __name__ == "__main__":
    main()
