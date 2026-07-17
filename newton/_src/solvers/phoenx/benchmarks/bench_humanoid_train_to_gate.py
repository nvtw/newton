# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure PhoenX Humanoid time from zero training to a physical walking gate."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training.examples import train_humanoid_phoenx_ppo as humanoid


@dataclass(frozen=True)
class StatsEvaluateHumanoid:
    """Deterministic no-reset Humanoid locomotion metrics."""

    steps: int
    fall_fraction: float
    survival_fraction: float
    mean_forward_velocity: float
    mean_displacement_x: float
    forward_fraction: float
    mean_success: float
    mean_upright_cos: float
    mean_action_rms: float
    finite: bool


@wp.kernel(enable_backward=False)
def _accumulate_humanoid_gate_kernel(
    obs: wp.array2d[wp.float32],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    actions: wp.array2d[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    alive: wp.array[wp.int32],
    survival_steps: wp.array[wp.int32],
    forward_sum: wp.array[wp.float32],
    success_sum: wp.array[wp.float32],
    upright_sum: wp.array[wp.float32],
    action_sq_sum: wp.array[wp.float32],
    last_alive_x: wp.array[wp.float32],
    finite: wp.array[wp.int32],
):
    world = wp.tid()
    if alive[world] == wp.int32(0):
        return
    q_base = world * coord_stride
    qd_base = world * dof_stride
    qx = joint_q[q_base + wp.int32(3)]
    qy = joint_q[q_base + wp.int32(4)]
    forward_sum[world] = forward_sum[world] + joint_qd[qd_base]
    success_sum[world] = success_sum[world] + successes[world]
    upright_sum[world] = upright_sum[world] + wp.float32(1.0) - wp.float32(2.0) * (qx * qx + qy * qy)
    last_alive_x[world] = joint_q[q_base]
    survival_steps[world] = survival_steps[world] + wp.int32(1)
    for action in range(21):
        value = actions[world, action]
        action_sq_sum[world] = action_sq_sum[world] + value * value
    for col in range(75):
        if not wp.isfinite(obs[world, col]):
            finite[world] = wp.int32(0)
    if dones[world] > wp.float32(0.5):
        alive[world] = wp.int32(0)


def evaluate_humanoid(
    trainer: rl.TrainerPPO,
    args: argparse.Namespace,
    *,
    gate_seed: int,
) -> StatsEvaluateHumanoid:
    env = humanoid.make_env(args, world_count=int(args.eval_world_count), auto_reset=False)
    obs = env.reset()
    trainer.reset_rollout_state()
    initial_q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
    start_x = initial_q[:, 0].copy()
    alive = wp.ones(env.world_count, dtype=wp.int32, device=env.device)
    survival_steps = wp.zeros(env.world_count, dtype=wp.int32, device=env.device)
    forward_sum = wp.zeros(env.world_count, dtype=wp.float32, device=env.device)
    success_sum = wp.zeros(env.world_count, dtype=wp.float32, device=env.device)
    upright_sum = wp.zeros(env.world_count, dtype=wp.float32, device=env.device)
    action_sq_sum = wp.zeros(env.world_count, dtype=wp.float32, device=env.device)
    last_alive_x = wp.array(start_x, dtype=wp.float32, device=env.device)
    finite = wp.ones(env.world_count, dtype=wp.int32, device=env.device)

    with wp.ScopedCapture(device=env.device) as capture:
        actions, _log_probs, _values = trainer.act(obs, seed=gate_seed, deterministic=True)
        env.step(actions)
        wp.launch(
            _accumulate_humanoid_gate_kernel,
            dim=env.world_count,
            inputs=[
                env.obs,
                env.state_0.joint_q,
                env.state_0.joint_qd,
                actions,
                env.step_dones,
                env.step_successes,
                env.coord_stride,
                env.dof_stride,
            ],
            outputs=[
                alive,
                survival_steps,
                forward_sum,
                success_sum,
                upright_sum,
                action_sq_sum,
                last_alive_x,
                finite,
            ],
            device=env.device,
        )
    for _ in range(int(args.eval_steps)):
        wp.capture_launch(capture.graph)

    survival = survival_steps.numpy()
    sample_count = int(np.sum(survival))
    displacement = last_alive_x.numpy() - start_x
    return StatsEvaluateHumanoid(
        steps=int(args.eval_steps),
        fall_fraction=float(np.mean(alive.numpy() == 0)),
        survival_fraction=float(np.mean(survival)) / float(max(int(args.eval_steps), 1)),
        mean_forward_velocity=float(np.sum(forward_sum.numpy())) / float(max(sample_count, 1)),
        mean_displacement_x=float(np.mean(displacement)),
        forward_fraction=float(np.mean(displacement >= float(args.gate_min_individual_displacement))),
        mean_success=float(np.sum(success_sum.numpy())) / float(max(sample_count, 1)),
        mean_upright_cos=float(np.sum(upright_sum.numpy())) / float(max(sample_count, 1)),
        mean_action_rms=math.sqrt(float(np.sum(action_sq_sum.numpy())) / float(max(sample_count * env.action_dim, 1))),
        finite=bool(np.all(finite.numpy() != 0)),
    )


def check_gate(stats: StatsEvaluateHumanoid, args: argparse.Namespace) -> list[str]:
    checks = (
        (stats.finite, "state or action became non-finite"),
        (stats.fall_fraction <= float(args.gate_max_fall_fraction), "fall fraction"),
        (stats.survival_fraction >= float(args.gate_min_survival_fraction), "survival fraction"),
        (stats.mean_forward_velocity >= float(args.gate_min_forward_velocity), "mean forward velocity"),
        (stats.mean_displacement_x >= float(args.gate_min_displacement), "mean forward displacement"),
        (stats.forward_fraction >= float(args.gate_min_forward_fraction), "forward-moving fraction"),
        (stats.mean_success >= float(args.gate_min_success), "locomotion success"),
        (stats.mean_upright_cos >= float(args.gate_min_upright_cos), "mean upright cosine"),
    )
    return [label for passed, label in checks if not passed]


def benchmark_train_to_gate(args: argparse.Namespace) -> dict[str, Any]:
    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("Humanoid time-to-policy requires CUDA with Warp memory pooling")
    if int(args.iterations) < 0 or int(args.chunk_iterations) <= 0:
        raise ValueError("iterations must be nonnegative and chunk_iterations must be positive")
    if int(args.required_consecutive_passes) <= 0:
        raise ValueError("required_consecutive_passes must be positive")

    total_t0 = time.perf_counter()
    env = humanoid.make_env(args)
    sample_count = env.world_count * int(args.rollout_steps)
    ppo_config = humanoid.build_ppo_config(args, sample_count)
    trainer = humanoid.make_trainer(args, env, ppo_config)
    buffer = rl.BufferRollout(
        num_steps=int(args.rollout_steps),
        num_envs=env.world_count,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )
    start_iteration = int(trainer.iteration)
    completed_iterations = start_iteration
    gate_history: list[dict[str, Any]] = []
    first_pass: dict[str, Any] | None = None

    def evaluate_iteration(iteration: int) -> dict[str, Any] | None:
        checkpoint_path = Path(str(args.checkpoint_path).format(iteration=iteration))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(checkpoint_path, iteration=iteration)
        pass_streak = 0
        latest: dict[str, Any] | None = None
        for window in range(int(args.required_consecutive_passes)):
            gate_seed = int(args.gate_seed) + window * 1_000_003
            stats = evaluate_humanoid(trainer, args, gate_seed=gate_seed)
            failures = check_gate(stats, args)
            passed = not failures
            pass_streak = pass_streak + 1 if passed else 0
            latest = {
                "iteration": iteration,
                "samples": iteration * sample_count,
                "checkpoint": str(checkpoint_path),
                "window": window,
                "gate_seed": gate_seed,
                "stats": asdict(stats),
                "failures": failures,
                "pass_gate": passed,
                "consecutive_passes": pass_streak,
                "qualified": pass_streak >= int(args.required_consecutive_passes),
                "total_wall_seconds": time.perf_counter() - total_t0,
            }
            gate_history.append(latest)
            if not passed:
                break
        return latest if latest is not None and latest["qualified"] else None

    first_pass = evaluate_iteration(completed_iterations)
    graph = None
    if first_pass is None and completed_iterations < start_iteration + int(args.iterations):
        trainer.reserve_update_buffers(buffer)
        seed_counter = rl.make_seed_counter(
            int(args.seed) + completed_iterations * int(args.rollout_steps), device=device
        )
        with wp.ScopedCapture(device=device) as capture:
            rl.collect_ppo_rollout_seed_counter(env, trainer, buffer, seed_counter=seed_counter)
            trainer.update(buffer, read_stats=False)
        graph = capture.graph

    final_iteration = start_iteration + int(args.iterations)
    while first_pass is None and completed_iterations < final_iteration:
        chunk = min(int(args.chunk_iterations), final_iteration - completed_iterations)
        chunk_t0 = time.perf_counter()
        for _ in range(chunk):
            assert graph is not None
            wp.capture_launch(graph)
        completed_iterations += chunk
        trainer.iteration = completed_iterations
        elapsed = max(time.perf_counter() - chunk_t0, 1.0e-12)
        print(
            f"iter={completed_iterations:04d} chunk_sps={chunk * sample_count / elapsed:,.0f} "
            f"wall={time.perf_counter() - total_t0:.1f}s"
        )
        first_pass = evaluate_iteration(completed_iterations)

    result = {
        "engine": "phoenx_humanoid_warp_ppo_train_to_gate",
        "metric": "fresh graph training plus deterministic no-reset walking gates",
        "seed": int(args.seed),
        "gate_seed": int(args.gate_seed),
        "world_count": int(args.world_count),
        "rollout_steps": int(args.rollout_steps),
        "sim_substeps": int(args.sim_substeps),
        "solver_iterations": int(args.solver_iterations),
        "velocity_iterations": int(args.velocity_iterations),
        "required_consecutive_passes": int(args.required_consecutive_passes),
        "start_iteration": start_iteration,
        "completed_iterations": completed_iterations,
        "pass_gate": first_pass is not None,
        "first_pass": first_pass,
        "gate_history": gate_history,
        "total_wall_seconds": time.perf_counter() - total_t0,
    }
    output_path = Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return result


def _make_parser() -> argparse.ArgumentParser:
    parser = humanoid._make_parser()
    parser.description = __doc__
    parser.add_argument("--chunk-iterations", type=int, default=25)
    parser.add_argument("--gate-seed", type=int, default=1000)
    parser.add_argument("--required-consecutive-passes", type=int, default=1)
    parser.add_argument("--eval-world-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=600)
    parser.add_argument("--gate-max-fall-fraction", type=float, default=0.05)
    parser.add_argument("--gate-min-survival-fraction", type=float, default=0.95)
    parser.add_argument("--gate-min-forward-velocity", type=float, default=0.4)
    parser.add_argument("--gate-min-displacement", type=float, default=3.0)
    parser.add_argument("--gate-min-individual-displacement", type=float, default=2.0)
    parser.add_argument("--gate-min-forward-fraction", type=float, default=0.9)
    parser.add_argument("--gate-min-success", type=float, default=0.4)
    parser.add_argument("--gate-min-upright-cos", type=float, default=0.8)
    parser.add_argument("--json-output", required=True)
    parser.set_defaults(checkpoint_path="/tmp/phoenx_humanoid_gate_{iteration:04d}.npz")
    return parser


def main() -> int:
    args = _make_parser().parse_args()
    result = benchmark_train_to_gate(args)
    return 0 if result["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
