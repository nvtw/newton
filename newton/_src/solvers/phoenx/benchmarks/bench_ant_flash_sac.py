# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure FlashSAC wall-to-quality on reusable locomotion tasks."""

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
from ..rl_training.dr_legs import ConfigEnvDrLegsPhoenX, EnvDrLegsPhoenX, default_dr_legs_flash_sac_config
from ..rl_training.flash_sac import BufferReplayFlashSAC, TrainerFlashSAC
from ..rl_training.flash_sac_autotune import ConfigFlashSACLRAutotune, ControllerFlashSACLRAutotune
from ..rl_training.flash_sac_autotune_evaluation import EvaluatorPairedFlashSAC, _bootstrap_ready
from ..rl_training.go2 import ConfigEnvGo2PhoenX, EnvGo2PhoenX, default_go2_flash_sac_config
from ..rl_training.h1 import ConfigEnvH1PhoenX, EnvH1PhoenX, default_h1_flash_sac_config
from ..rl_training.humanoid import ConfigEnvHumanoidPhoenX, EnvHumanoidPhoenX, default_humanoid_flash_sac_config

_TASK_DEFAULT_WORLD_COUNTS = {
    "ant": 2048,
    "dr_legs": 4096,
    "h1": 4096,
    "go2": 1024,
    "humanoid": 4096,
}

_EnvConfig = (
    ConfigEnvAntPhoenX | ConfigEnvDrLegsPhoenX | ConfigEnvGo2PhoenX | ConfigEnvH1PhoenX | ConfigEnvHumanoidPhoenX
)
_Env = EnvAntPhoenX | EnvDrLegsPhoenX | EnvGo2PhoenX | EnvH1PhoenX | EnvHumanoidPhoenX


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
    env: _Env,
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


def _make_env_config(args: argparse.Namespace, *, world_count: int) -> _EnvConfig:
    """Build one explicit task protocol shared by training and evaluation."""

    sim_substeps = getattr(args, "sim_substeps", None)
    forward_command = getattr(args, "forward_command", None)
    if args.task == "dr_legs":
        values = {"task": "walk", "world_count": int(world_count), "auto_reset": True}
        if sim_substeps is not None:
            values["sim_substeps"] = int(sim_substeps)
        return ConfigEnvDrLegsPhoenX(**values)
    if args.task == "go2":
        values = {
            "world_count": int(world_count),
            "articulation_mode": str(args.articulation_mode),
            "auto_reset": True,
            "reward_mode": "dense_command",
            "command": (0.8 if forward_command is None else float(forward_command), 0.0, 0.0, 0.0),
        }
        if sim_substeps is not None:
            values["sim_substeps"] = int(sim_substeps)
        return ConfigEnvGo2PhoenX(**values)
    common = {"world_count": int(world_count), "articulation_mode": str(args.articulation_mode), "auto_reset": True}
    if sim_substeps is not None:
        common["sim_substeps"] = int(sim_substeps)
    if args.task == "h1":
        return ConfigEnvH1PhoenX(
            **common,
            command=(0.8 if forward_command is None else float(forward_command), 0.0, 0.0),
            randomize_commands=False,
        )
    if args.task == "humanoid":
        return ConfigEnvHumanoidPhoenX(**common)
    return ConfigEnvAntPhoenX(
        **common,
        task_profile="mraksha",
    )


def _make_evaluator(
    sources: tuple[TrainerFlashSAC, TrainerFlashSAC],
    env_config: _EnvConfig,
    args: argparse.Namespace,
) -> EvaluatorPairedFlashSAC:
    """Capture deterministic no-reset locomotion evaluation."""

    eval_config = replace(
        env_config,
        world_count=int(args.evaluation_worlds),
        auto_reset=False,
        max_episode_steps=0,
    )
    env_type = {
        "ant": EnvAntPhoenX,
        "dr_legs": EnvDrLegsPhoenX,
        "go2": EnvGo2PhoenX,
        "humanoid": EnvHumanoidPhoenX,
        "h1": EnvH1PhoenX,
    }[args.task]
    envs = (env_type(eval_config, device=args.device), env_type(eval_config, device=args.device))

    def forward_velocity(env: _Env, _rewards: wp.array[wp.float32]) -> wp.array[wp.float32]:
        if isinstance(env, EnvHumanoidPhoenX):
            return env.step_successes
        return env.step_forward_velocities

    return EvaluatorPairedFlashSAC(
        sources,
        envs,
        horizon_steps=int(args.evaluation_horizon),
        seed=10_001 + int(args.seed) * 9973,
        metric_source=forward_velocity,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run one measured locomotion seed and return its quality history."""

    device = wp.get_device(args.device)
    world_count = _TASK_DEFAULT_WORLD_COUNTS[args.task] if args.world_count is None else int(args.world_count)
    env_config = _make_env_config(args, world_count=world_count)
    if args.task == "humanoid":
        config = default_humanoid_flash_sac_config()
        env_type = EnvHumanoidPhoenX
    elif args.task == "dr_legs":
        config = default_dr_legs_flash_sac_config()
        env_type = EnvDrLegsPhoenX
    elif args.task == "go2":
        config = default_go2_flash_sac_config()
        env_type = EnvGo2PhoenX
    elif args.task == "h1":
        config = default_h1_flash_sac_config()
        env_type = EnvH1PhoenX
    else:
        config = default_ant_flash_sac_config()
        env_type = EnvAntPhoenX
    if args.base_lr is not None:
        learning_rate = float(args.base_lr)
        config = replace(config, actor_lr=learning_rate, critic_lr=learning_rate, alpha_lr=learning_rate)
    setup_phases: dict[str, float] = {}
    setup_start = time.perf_counter()
    phase_start = setup_start
    env = env_type(env_config, device=device)
    setup_phases["environment"] = time.perf_counter() - phase_start
    print(f"setup environment: {setup_phases['environment']:.3f} s", flush=True)
    phase_start = time.perf_counter()
    champion = TrainerFlashSAC(
        obs_dim=env.obs_dim,
        action_dim=int(getattr(env, "policy_action_dim", env.action_dim)),
        config=config,
        device=device,
        seed=int(args.seed),
    )
    setup_phases["trainer"] = time.perf_counter() - phase_start
    print(f"setup trainer: {setup_phases['trainer']:.3f} s", flush=True)
    phase_start = time.perf_counter()
    replay = _warm_replay(champion, env, seed=31 + int(args.seed) * 9973)
    setup_phases["replay_warmup"] = time.perf_counter() - phase_start
    print(f"setup replay warmup: {setup_phases['replay_warmup']:.3f} s", flush=True)
    phase_start = time.perf_counter()
    controller = None
    if args.mode == "fixed":
        training = champion.capture_training_graph(
            env,
            replay,
            updates_per_step=int(args.updates_per_step),
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
                exploit_after_candidate=False,
                bootstrap_single_policy=not bool(args.disable_search_bootstrap),
                seed=int(args.seed),
            ),
        )
        training = controller.capture_overlap(
            env,
            replay,
            updates_per_step=int(args.updates_per_step),
            interactions_per_launch=2,
            seed=31 + int(args.seed) * 9973,
            population_backend="parallel",
        )
        if controller.config.bootstrap_single_policy:
            training.start_single_policy_bootstrap()
        sources = training.evaluation_trainers()
    setup_phases["training_capture"] = time.perf_counter() - phase_start
    print(f"setup training capture: {setup_phases['training_capture']:.3f} s", flush=True)
    phase_start = time.perf_counter()
    evaluator = _make_evaluator(sources, env_config, args)
    setup_phases["evaluator"] = time.perf_counter() - phase_start
    print(f"setup evaluator: {setup_phases['evaluator']:.3f} s", flush=True)
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
                evaluated_target_update_rates = [config.tau, config.tau]
                evaluated_policy_frequencies = [config.policy_frequency, config.policy_frequency]
                action = "fixed"
            else:
                eval_sources = training.evaluation_trainers()
                evaluated_rates = controller.member_rates.tolist()
                evaluated_target_update_rates = controller.member_target_update_rates.tolist()
                evaluated_policy_frequencies = controller.member_policy_frequencies.tolist()
            scores = evaluator.evaluate(eval_sources)
            mean_velocity = float(np.mean(scores.champion_scores))
            passed = bool(
                scores.champion_finite
                and scores.champion_termination_rate <= float(args.maximum_termination_rate)
                and mean_velocity >= float(args.minimum_forward_velocity)
            )
            consecutive_passes = consecutive_passes + 1 if passed else 0
            if controller is not None and controller.bootstrapping:
                if _bootstrap_ready(
                    mean_velocity,
                    float(scores.champion_termination_rate),
                    bool(scores.champion_finite),
                    float(controller.config.informative_score_threshold),
                    float(controller.config.bootstrap_max_termination_rate),
                ):
                    training.reopen_search()
                    action = "start_search"
                else:
                    action = "bootstrap"
            elif controller is not None and not controller.converged:
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
                    "evaluated_target_update_rates": evaluated_target_update_rates,
                    "evaluated_policy_frequencies": evaluated_policy_frequencies,
                    "next_target_update_rates": controller.member_target_update_rates.tolist()
                    if controller is not None
                    else evaluated_target_update_rates,
                    "next_policy_frequencies": controller.member_policy_frequencies.tolist()
                    if controller is not None
                    else evaluated_policy_frequencies,
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
        "task": args.task,
        "world_count": env.world_count,
        "articulation_mode": args.articulation_mode,
        "sim_substeps": env_config.sim_substeps,
        "updates_per_step": int(args.updates_per_step),
        "setup_seconds": setup_seconds,
        "setup_phases": setup_phases,
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
    parser.add_argument("--task", choices=("ant", "dr_legs", "go2", "h1", "humanoid"), default="ant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world-count", type=int, default=None)
    parser.add_argument(
        "--articulation-mode",
        choices=("maximal", "maximal_projected", "maximal_articulated", "hybrid", "reduced"),
        default="reduced",
    )
    parser.add_argument("--sim-substeps", type=int, default=None)
    parser.add_argument("--max-launches", type=int, default=3000)
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--forward-command", type=float, default=None)
    parser.add_argument(
        "--disable-search-bootstrap",
        action="store_true",
        help="Start both policies immediately for active-search profiling.",
    )
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
