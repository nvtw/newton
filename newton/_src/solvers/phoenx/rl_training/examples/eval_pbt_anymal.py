# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate whether PBT beats a fixed-hparam PPO baseline on Anymal walking.

Both runs share the same fixed task (forward walking at 0.6 m/s, dense command
reward, no curriculum/randomization) so the only variable is hyper-parameter
adaptation. The baseline trains one worker with the default starting hparams
for ``--iters`` iterations; PBT runs ``--pop`` workers for the same number of
per-worker iterations and adapts actor_lr / entropy_coeff / max_grad_norm via
truncation selection.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import newton.rl as rl

# Fixed-task env: matches the production Anymal forward-walking reward shaping
# (train_anymal_walk_phoenx_ppo._BASE_ENV) but with a fixed 0.6 m/s command so
# the task is stationary across PBT cycles and graph_leapfrog-safe.
_BASE_ENV = {
    "reward_mode": "dense_command",
    "command": (0.6, 0.0, 0.0, 0.0),
    "max_episode_steps": 500,
    "lin_vel_reward_scale": 1.0,
    "yaw_rate_reward_scale": 0.5,
    "lin_vel_tracking_sigma": 0.5,
    "yaw_rate_tracking_sigma": 0.5,
    "base_height_reward_scale": 0.75,
    "base_height_tracking_sigma": 0.06,
    "z_vel_reward_scale": -2.0,
    "ang_vel_reward_scale": -0.05,
    "action_rate_reward_scale": -0.01,
    "joint_speed_reward_scale": -1.0e-4,
    "flat_orientation_reward_scale": -5.0,
    "fall_reward_scale": -2.0,
    "energy_reward_scale": -2.5e-5,
    "hip_abduction_reward_scale": -0.15,
    "min_base_height": 0.30,
    "min_upright_cos": 0.35,
    "action_scale": 0.5,
    "target_base_height": 0.62,
    "actuator_ke": 150.0,
    "actuator_kd": 5.0,
}


def _env_config(world_count: int) -> rl.ConfigEnvAnymalPhoenX:
    return rl.ConfigEnvAnymalPhoenX(world_count=int(world_count), **_BASE_ENV)


def _initial_ppo_config() -> rl.ConfigPPO:
    # Deliberately a generic, untuned starting point shared by both the
    # baseline and every PBT worker before perturbation.
    return rl.ConfigPPO(
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coeff=5.0e-3,
        actor_lr=1.0e-3,
        critic_lr=1.0e-3,
        train_epochs=5,
        normalize_advantages=True,
        max_grad_norm=1.0,
        mirror_loss_coeff=0.0,
    )


def _base_train_config(args: argparse.Namespace, iterations: int) -> rl.ConfigTrainAnymalPPO:
    return rl.ConfigTrainAnymalPPO(
        iterations=int(iterations),
        rollout_steps=int(args.rollout_steps),
        hidden_layers=(128, 128, 128),
        activation="elu",
        log_std_init=0.0,
        env_config=_env_config(args.world_count),
        ppo_config=_initial_ppo_config(),
        device=args.device,
        seed=int(args.seed),
        log_interval=0,
        use_target_curriculum=False,
        randomize_target_positions=False,
        randomize_commands=False,
        execution_mode=str(args.execution_mode),
    )


def _mean_tail_reward(history: list, window: int) -> float:
    if not history:
        return float("nan")
    tail = history[-max(1, window) :]
    return float(np.mean([s.mean_reward for s in tail]))


def run_baseline(args: argparse.Namespace) -> float:
    iters = int(args.pop_cycles) * int(args.cycle_iters)
    print(f"\n=== BASELINE PPO: {iters} iters, default hparams ===", flush=True)
    t0 = time.perf_counter()
    result = rl.train_anymal_ppo(_base_train_config(args, iters))
    dt = time.perf_counter() - t0
    fit = _mean_tail_reward(result.history, args.fitness_window)
    print(f"baseline: final mean_reward (tail{args.fitness_window})={fit:.4f}  [{dt:.1f}s]", flush=True)
    return fit


def run_pbt(args: argparse.Namespace) -> tuple[float, float]:
    print(
        f"\n=== PBT: pop={args.pop} cycles={args.pop_cycles} cycle_iters={args.cycle_iters} "
        f"({args.pop_cycles * args.cycle_iters} iters/worker) ===",
        flush=True,
    )
    pbt_config = rl.ConfigPBT(
        population_size=int(args.pop),
        exploit_interval=int(args.cycle_iters),
        total_cycles=int(args.pop_cycles),
        exploit_fraction=0.25,
        fitness_window=int(args.fitness_window),
        fitness_metric="mean_reward",
        exploit_strategy="truncation",
        fresh_optimizer_on_exploit=True,
        seed=int(args.seed),
        log_interval=1,
    )
    specs = [
        rl.HparamSpec("actor_lr", "log_uniform", 1.0e-4, 5.0e-3),
        rl.HparamSpec("critic_lr", "log_uniform", 1.0e-4, 5.0e-3),
        rl.HparamSpec("entropy_coeff", "log_uniform", 1.0e-5, 2.0e-2),
        rl.HparamSpec("max_grad_norm", "log_uniform", 0.2, 2.0),
    ]
    t0 = time.perf_counter()
    result = rl.population_based_train_anymal(
        _base_train_config(args, args.cycle_iters),
        pbt_config=pbt_config,
        hparam_specs=specs,
        output_dir=str(args.output_dir),
    )
    dt = time.perf_counter() - t0

    # Headline metric: best final-cycle reward across workers (the fully-trained
    # policies). result.best_fitness uses a running-mean over cycles, so it lags
    # the actual end-of-training performance and is reported only for reference.
    def _last(w) -> float:
        return float(w.fitness_history[-1]) if w.fitness_history else float("-inf")

    def _peak(w) -> float:
        return float(max(w.fitness_history)) if w.fitness_history else float("-inf")

    best_final = max(_last(w) for w in result.workers)
    best_peak = max(_peak(w) for w in result.workers)
    print(
        f"pbt: best_final_cycle={best_final:.4f} best_peak_cycle={best_peak:.4f} "
        f"running_mean_best={result.best_fitness:.4f} (w{result.best_worker_id})  [{dt:.1f}s]",
        flush=True,
    )
    for w in sorted(result.workers, key=lambda x: x.worker_id):
        hp = w.hparam_history[-1] if w.hparam_history else {}
        print(
            f"  w{w.worker_id}: final={_last(w):.4f} peak={_peak(w):.4f} gen={w.generation} "
            f"actor_lr={hp.get('actor_lr', float('nan')):.2e} "
            f"entropy={hp.get('entropy_coeff', float('nan')):.2e} "
            f"grad_norm={hp.get('max_grad_norm', float('nan')):.2f}",
            flush=True,
        )
    return float(best_final), float(best_peak)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--world-count", type=int, default=512, dest="world_count")
    p.add_argument("--rollout-steps", type=int, default=32, dest="rollout_steps")
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--pop-cycles", type=int, default=8, dest="pop_cycles")
    p.add_argument("--cycle-iters", type=int, default=20, dest="cycle_iters")
    p.add_argument("--fitness-window", type=int, default=8, dest="fitness_window")
    p.add_argument("--execution-mode", choices=("eager", "graph_leapfrog"), default="graph_leapfrog")
    p.add_argument("--output-dir", default="/tmp/pbt_anymal")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-pbt", action="store_true")
    args = p.parse_args(argv)

    baseline_fit = float("nan")
    if not args.skip_baseline:
        baseline_fit = run_baseline(args)

    pbt_final = pbt_peak = float("nan")
    if not args.skip_pbt:
        pbt_final, pbt_peak = run_pbt(args)

    print("\n=== SUMMARY ===", flush=True)
    print(f"baseline final reward    : {baseline_fit:.4f}", flush=True)
    print(f"pbt best final-cycle     : {pbt_final:.4f}", flush=True)
    print(f"pbt best peak-cycle      : {pbt_peak:.4f}", flush=True)
    if np.isfinite(baseline_fit) and np.isfinite(pbt_final):
        delta = pbt_final - baseline_fit
        pct = 100.0 * delta / (abs(baseline_fit) + 1e-9)
        verdict = "PBT WINS" if delta > 0 else "baseline >= PBT"
        print(f"delta (pbt-baseline)  : {delta:+.4f} ({pct:+.1f}%)  -> {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
