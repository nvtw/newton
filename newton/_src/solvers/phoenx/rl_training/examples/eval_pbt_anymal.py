# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate whether PBT beats a fixed-hparam PPO baseline on Anymal walking.

Both runs share the same fixed task (forward walking at 0.6 m/s, dense command
reward, no curriculum/randomization) so the only variable is hyper-parameter
adaptation. The baseline trains one worker with the default starting hparams
for ``--iters`` iterations; PBT runs ``--pop`` workers for the same number of
per-worker iterations and adapts actor_lr / entropy_coeff / max_grad_norm via
truncation selection.

After training, both checkpoints are scored with a no-reset walking rollout
(``auto_reset`` off, no episode-time limit) so falls are not hidden by rollout
resets. This measures gait quality -- fall fraction, survival, forward-velocity
tracking, distance walked -- rather than just training reward.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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


def run_baseline(args: argparse.Namespace) -> tuple[float, str]:
    iters = int(args.pop_cycles) * int(args.cycle_iters)
    print(f"\n=== BASELINE PPO: {iters} iters, default hparams ===", flush=True)
    t0 = time.perf_counter()
    result = rl.train_anymal_ppo(_base_train_config(args, iters))
    dt = time.perf_counter() - t0
    fit = _mean_tail_reward(result.history, args.fitness_window)
    ckpt = str(Path(args.output_dir) / "baseline.npz")
    result.trainer.save_checkpoint(ckpt, iteration=int(result.trainer.iteration))
    print(f"baseline: final mean_reward (tail{args.fitness_window})={fit:.4f}  [{dt:.1f}s]", flush=True)
    return fit, ckpt


def run_pbt(args: argparse.Namespace) -> tuple[float, float, str]:
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
    return float(best_final), float(best_peak), str(result.best_checkpoint)


def evaluate_walk(env: rl.EnvAnymalPhoenX, ckpt_path: str, args: argparse.Namespace, *, label: str) -> dict:
    """Score a checkpoint with a no-reset forward-walk rollout.

    Falls are permanent (``auto_reset`` is off on *env*); per-world metrics are
    accumulated only while a world is still upright.
    """
    trainer = rl.load_ppo_checkpoint(ckpt_path, device=args.device)
    obs = env.reset()
    trainer.reset_rollout_state()
    wc = env.world_count
    cmd_x = float(_BASE_ENV["command"][0])

    first_done = np.full(wc, -1, dtype=np.int32)
    prev_xy = env.state_0.joint_q.numpy().reshape(wc, env.coord_stride)[:, 0:2].copy()
    # Path length is heading-agnostic: the command is body-frame vx, so a policy
    # that walks fast while curving covers ground without net world-+x progress.
    path_len = np.zeros(wc, dtype=np.float64)
    fwd_vel_sum = 0.0
    vx_err_sum = 0.0
    alive_steps = 0

    for step in range(int(args.eval_steps)):
        alive = first_done < 0
        actions, _, _ = trainer.act(obs, seed=int(args.seed) + 90_000 + step, deterministic=True)
        obs, _rewards, dones = env.step(actions)
        done_np = dones.numpy() > 0.5
        obs_np = obs.numpy()
        xy = env.state_0.joint_q.numpy().reshape(wc, env.coord_stride)[:, 0:2]
        if np.any(alive):
            lin = obs_np[alive, 0:2]
            fwd_vel_sum += float(np.sum(lin[:, 0]))
            vx_err_sum += float(np.sum(np.abs(lin[:, 0] - cmd_x)))
            path_len[alive] += np.linalg.norm(xy[alive] - prev_xy[alive], axis=1)
            alive_steps += int(np.sum(alive))
        prev_xy = xy.copy()
        first_done[(first_done < 0) & done_np] = step + 1

    survival = np.where(first_done >= 0, first_done, int(args.eval_steps))
    den = float(max(alive_steps, 1))
    metrics = {
        "fall_fraction": float(np.mean(first_done >= 0)),
        "survival_fraction": float(np.mean(survival)) / float(args.eval_steps),
        "mean_forward_velocity": fwd_vel_sum / den,
        "mean_abs_vx_error": vx_err_sum / den,
        "mean_path_length": float(np.mean(path_len)),
    }
    print(
        f"  {label:8s}: fall={metrics['fall_fraction']:.3f} survival={metrics['survival_fraction']:.3f} "
        f"vx={metrics['mean_forward_velocity']:.3f} (cmd {cmd_x:.2f}) "
        f"|vx-cmd|={metrics['mean_abs_vx_error']:.3f} path_len={metrics['mean_path_length']:.3f}m",
        flush=True,
    )
    return metrics


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
    p.add_argument("--eval-world-count", type=int, default=256, dest="eval_world_count")
    p.add_argument("--eval-steps", type=int, default=400, dest="eval_steps")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-pbt", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args(argv)

    baseline_fit = float("nan")
    baseline_ckpt = None
    if not args.skip_baseline:
        baseline_fit, baseline_ckpt = run_baseline(args)

    pbt_final = pbt_peak = float("nan")
    pbt_ckpt = None
    if not args.skip_pbt:
        pbt_final, pbt_peak, pbt_ckpt = run_pbt(args)

    baseline_walk = pbt_walk = None
    if not args.skip_eval and (baseline_ckpt or pbt_ckpt):
        print(
            f"\n=== NO-RESET WALK EVAL: {args.eval_world_count} worlds x {args.eval_steps} steps ===",
            flush=True,
        )
        eval_env = rl.EnvAnymalPhoenX(
            rl.ConfigEnvAnymalPhoenX(
                world_count=int(args.eval_world_count), auto_reset=False, **{**_BASE_ENV, "max_episode_steps": 0}
            ),
            device=args.device,
        )
        if baseline_ckpt:
            baseline_walk = evaluate_walk(eval_env, baseline_ckpt, args, label="baseline")
        if pbt_ckpt:
            pbt_walk = evaluate_walk(eval_env, pbt_ckpt, args, label="pbt-best")

    print("\n=== SUMMARY ===", flush=True)
    print(f"baseline final reward    : {baseline_fit:.4f}", flush=True)
    print(f"pbt best final-cycle     : {pbt_final:.4f}", flush=True)
    print(f"pbt best peak-cycle      : {pbt_peak:.4f}", flush=True)
    if np.isfinite(baseline_fit) and np.isfinite(pbt_final):
        delta = pbt_final - baseline_fit
        pct = 100.0 * delta / (abs(baseline_fit) + 1e-9)
        verdict = "PBT WINS" if delta > 0 else "baseline >= PBT"
        print(f"train-reward delta       : {delta:+.4f} ({pct:+.1f}%)  -> {verdict}", flush=True)
    if baseline_walk and pbt_walk:
        print("\ngait quality (no-reset):       baseline     pbt-best     delta", flush=True)
        for key, better in (
            ("fall_fraction", "lower"),
            ("survival_fraction", "higher"),
            ("mean_forward_velocity", "higher"),
            ("mean_abs_vx_error", "lower"),
            ("mean_path_length", "higher"),
        ):
            b, pv = baseline_walk[key], pbt_walk[key]
            d = pv - b
            win = (d > 0) if better == "higher" else (d < 0)
            print(f"  {key:24s} {b:10.3f} {pv:12.3f} {d:+10.3f}  ({'PBT' if win else 'base'} better)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
