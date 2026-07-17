# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure PhoenX DR Legs hold-pose time to a physically qualified policy."""

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


@wp.kernel(enable_backward=False)
def _apply_translation_kick_kernel(
    body_stride: wp.int32,
    seed: wp.int32,
    max_speed: wp.float32,
    body_qd: wp.array[wp.spatial_vector],
):
    world, body_local = wp.tid()
    rng = wp.rand_init(seed, world * wp.int32(747796405) + wp.int32(289133645))
    angle = wp.float32(2.0) * wp.pi * wp.randf(rng)
    radius = max_speed * wp.sqrt(wp.randf(rng))
    delta = wp.vec3(radius * wp.cos(angle), radius * wp.sin(angle), wp.float32(0.0))
    body = world * body_stride + body_local
    velocity = body_qd[body]
    body_qd[body] = wp.spatial_vector(wp.spatial_top(velocity) + delta, wp.spatial_bottom(velocity))


@wp.kernel(enable_backward=False)
def _accumulate_hold_gate_kernel(
    body_q: wp.array[wp.transform],
    actions: wp.array2d[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
    initial_xy: wp.array2d[wp.float32],
    body_stride: wp.int32,
    alive: wp.array[wp.int32],
    survival_steps: wp.array[wp.int32],
    success_sum: wp.array[wp.float32],
    min_height: wp.array[wp.float32],
    max_height: wp.array[wp.float32],
    min_upright: wp.array[wp.float32],
    max_drift: wp.array[wp.float32],
    action_sq_sum: wp.array[wp.float32],
    finite: wp.array[wp.int32],
):
    world = wp.tid()
    if alive[world] == wp.int32(0):
        return
    pelvis = body_q[world * body_stride]
    position = wp.transform_get_translation(pelvis)
    rotation = wp.transform_get_rotation(pelvis)
    upright = wp.float32(1.0) - wp.float32(2.0) * (rotation[0] * rotation[0] + rotation[1] * rotation[1])
    dx = position[0] - initial_xy[world, 0]
    dy = position[1] - initial_xy[world, 1]
    drift = wp.sqrt(dx * dx + dy * dy)
    success_sum[world] = success_sum[world] + successes[world]
    min_height[world] = wp.min(min_height[world], position[2])
    max_height[world] = wp.max(max_height[world], position[2])
    min_upright[world] = wp.min(min_upright[world], upright)
    max_drift[world] = wp.max(max_drift[world], drift)
    survival_steps[world] = survival_steps[world] + wp.int32(1)

    action = wp.int32(0)
    while action < actions.shape[1]:
        value = actions[world, action]
        action_sq_sum[world] = action_sq_sum[world] + value * value
        if not wp.isfinite(value):
            finite[world] = wp.int32(0)
        action = action + wp.int32(1)
    body_local = wp.int32(0)
    while body_local < body_stride:
        transform = body_q[world * body_stride + body_local]
        body_position = wp.transform_get_translation(transform)
        body_rotation = wp.transform_get_rotation(transform)
        if (
            not wp.isfinite(body_position[0])
            or not wp.isfinite(body_position[1])
            or not wp.isfinite(body_position[2])
            or not wp.isfinite(body_rotation[0])
            or not wp.isfinite(body_rotation[1])
            or not wp.isfinite(body_rotation[2])
            or not wp.isfinite(body_rotation[3])
        ):
            finite[world] = wp.int32(0)
        body_local = body_local + wp.int32(1)
    if dones[world] > wp.float32(0.5):
        alive[world] = wp.int32(0)


@dataclass(frozen=True)
class StatsEvaluateDrLegsHold:
    """No-reset physical hold-pose metrics."""

    steps: int
    fall_fraction: float
    survival_fraction: float
    mean_success: float
    min_pelvis_height: float
    max_pelvis_height: float
    min_upright_cos: float
    max_horizontal_drift: float
    mean_action_rms: float
    max_anchor_residual: float
    finite: bool


def _env_config(
    args: argparse.Namespace,
    *,
    world_count: int,
    auto_reset: bool,
    command_seed: int,
) -> rl.ConfigEnvDrLegsPhoenX:
    return rl.ConfigEnvDrLegsPhoenX(
        task="hold",
        world_count=world_count,
        sim_substeps=int(args.sim_substeps),
        collision_refresh_interval=int(args.collision_refresh_interval),
        solver_iterations=int(args.solver_iterations),
        velocity_iterations=int(args.velocity_iterations),
        max_episode_steps=int(args.max_episode_steps) if auto_reset else 0,
        command_seed=command_seed,
        auto_reset=auto_reset,
    )


def _ppo_config(args: argparse.Namespace, sample_count: int) -> rl.ConfigPPO:
    return rl.ConfigPPO(
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coeff=0.001,
        value_loss_coeff=1.0,
        actor_lr=float(args.learning_rate),
        critic_lr=float(args.learning_rate),
        train_epochs=int(args.train_epochs),
        minibatch_size=int(args.minibatch_size) if int(args.minibatch_size) > 0 else max(sample_count // 4, 1),
        normalize_advantages=True,
        max_grad_norm=1.0,
        manual_actor_backward=True,
        manual_critic_backward=True,
    )


def _quat_rotate(q: np.ndarray, value: np.ndarray) -> np.ndarray:
    vector = q[..., :3]
    return value + 2.0 * q[..., 3:4] * np.cross(vector, value) + 2.0 * np.cross(vector, np.cross(vector, value))


def _max_anchor_residual(env: rl.EnvDrLegsPhoenX) -> float:
    body_q = env.state_0.body_q.numpy()
    parent = env.model.joint_parent.numpy()
    child = env.model.joint_child.numpy()
    parent_xform = env.model.joint_X_p.numpy()
    child_xform = env.model.joint_X_c.numpy()
    parent_anchor = body_q[parent, :3] + _quat_rotate(body_q[parent, 3:], parent_xform[:, :3])
    child_anchor = body_q[child, :3] + _quat_rotate(body_q[child, 3:], child_xform[:, :3])
    return float(np.max(np.linalg.norm(parent_anchor - child_anchor, axis=1)))


def evaluate_hold(
    trainer: rl.TrainerPPO,
    args: argparse.Namespace,
    *,
    gate_seed: int,
) -> StatsEvaluateDrLegsHold:
    env = rl.EnvDrLegsPhoenX(
        _env_config(
            args,
            world_count=int(args.eval_world_count),
            auto_reset=False,
            command_seed=gate_seed,
        ),
        device=args.device,
    )
    obs = env.reset()
    wp.launch(
        _apply_translation_kick_kernel,
        dim=(env.world_count, env.body_stride),
        inputs=[env.body_stride, gate_seed, float(args.eval_kick_speed)],
        outputs=[env.state_0.body_qd],
        device=env.device,
    )
    trainer.reset_rollout_state()
    initial_q = env.state_0.body_q.numpy().reshape(env.world_count, env.body_stride, 7)
    initial_xy_np = initial_q[:, 0, :2].copy()
    initial_xy = wp.array(initial_xy_np, dtype=wp.float32, device=env.device)
    alive = wp.ones(env.world_count, dtype=wp.int32, device=env.device)
    survival_steps = wp.zeros(env.world_count, dtype=wp.int32, device=env.device)
    success_sum = wp.zeros(env.world_count, dtype=wp.float32, device=env.device)
    min_height = wp.full(env.world_count, math.inf, dtype=wp.float32, device=env.device)
    max_height = wp.full(env.world_count, -math.inf, dtype=wp.float32, device=env.device)
    min_upright = wp.full(env.world_count, math.inf, dtype=wp.float32, device=env.device)
    max_drift = wp.zeros(env.world_count, dtype=wp.float32, device=env.device)
    action_sq_sum = wp.zeros(env.world_count, dtype=wp.float32, device=env.device)
    finite = wp.ones(env.world_count, dtype=wp.int32, device=env.device)

    trainer.reserve_buffers(env.world_count)
    graphs = []
    graph_count = 2 if int(env.config.sim_substeps) % 2 else 1
    for _ in range(graph_count):
        with wp.ScopedCapture(device=env.device) as capture:
            actions, _log_probs, _values = trainer.act_reuse(obs, seed=gate_seed, deterministic=True)
            env.step(actions)
            wp.launch(
                _accumulate_hold_gate_kernel,
                dim=env.world_count,
                inputs=[
                    env.state_0.body_q,
                    actions,
                    env.step_dones,
                    env.step_successes,
                    initial_xy,
                    env.body_stride,
                ],
                outputs=[
                    alive,
                    survival_steps,
                    success_sum,
                    min_height,
                    max_height,
                    min_upright,
                    max_drift,
                    action_sq_sum,
                    finite,
                ],
                device=env.device,
            )
        graphs.append(capture.graph)
    for step in range(int(args.eval_steps)):
        wp.capture_launch(graphs[step % graph_count])

    survival = survival_steps.numpy()
    alive_samples = int(np.sum(survival))
    return StatsEvaluateDrLegsHold(
        steps=int(args.eval_steps),
        fall_fraction=float(np.mean(alive.numpy() == 0)),
        survival_fraction=float(np.mean(survival)) / float(max(int(args.eval_steps), 1)),
        mean_success=float(np.sum(success_sum.numpy())) / float(max(alive_samples, 1)),
        min_pelvis_height=float(np.min(min_height.numpy())),
        max_pelvis_height=float(np.max(max_height.numpy())),
        min_upright_cos=float(np.min(min_upright.numpy())),
        max_horizontal_drift=float(np.max(max_drift.numpy())),
        mean_action_rms=math.sqrt(float(np.sum(action_sq_sum.numpy())) / float(max(alive_samples * env.action_dim, 1))),
        max_anchor_residual=_max_anchor_residual(env),
        finite=bool(np.all(finite.numpy() != 0)),
    )


def check_gate(stats: StatsEvaluateDrLegsHold, args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    checks = (
        (stats.finite, "state or action became non-finite"),
        (stats.fall_fraction <= float(args.gate_max_fall_fraction), "fall fraction"),
        (stats.survival_fraction >= float(args.gate_min_survival_fraction), "survival fraction"),
        (stats.mean_success >= float(args.gate_min_success), "hold success"),
        (stats.min_pelvis_height >= float(args.gate_min_pelvis_height), "minimum pelvis height"),
        (stats.max_pelvis_height <= float(args.gate_max_pelvis_height), "maximum pelvis height"),
        (stats.min_upright_cos >= float(args.gate_min_upright_cos), "minimum upright cosine"),
        (stats.max_horizontal_drift <= float(args.gate_max_horizontal_drift), "horizontal drift"),
        (stats.max_anchor_residual <= float(args.gate_max_anchor_residual), "closed-loop anchor residual"),
    )
    for passed, label in checks:
        if not passed:
            failures.append(label)
    return failures


def _checkpoint_path(template: str, iteration: int) -> Path:
    return Path(template.format(iteration=iteration))


def benchmark_train_to_gate(args: argparse.Namespace) -> dict[str, Any]:
    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("DR Legs time-to-policy requires CUDA with Warp memory pooling")
    if int(args.iterations) <= 0 or int(args.chunk_iterations) <= 0:
        raise ValueError("iterations and chunk_iterations must be positive")
    if int(args.required_consecutive_passes) <= 0:
        raise ValueError("required_consecutive_passes must be positive")

    env = rl.EnvDrLegsPhoenX(
        _env_config(args, world_count=int(args.world_count), auto_reset=True, command_seed=int(args.seed)),
        device=device,
    )
    sample_count = env.world_count * int(args.rollout_steps)
    ppo_config = _ppo_config(args, sample_count)
    trainer = rl.TrainerPPO(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_layers=tuple(int(width) for width in args.hidden_layers),
        config=ppo_config,
        device=device,
        seed=int(args.seed),
        squash_actions=True,
        activation="elu",
        log_std_init=0.0,
    )
    buffer = rl.BufferRollout(
        num_steps=int(args.rollout_steps),
        num_envs=env.world_count,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )
    total_t0 = time.perf_counter()
    gate_history: list[dict[str, Any]] = []
    first_pass: dict[str, Any] | None = None
    pass_streak = 0
    completed_iterations = 0
    gate_index = 0
    checkpoint_template = str(args.checkpoint_path)

    def evaluate_iteration(iteration: int) -> dict[str, Any]:
        nonlocal gate_index, pass_streak
        checkpoint = _checkpoint_path(checkpoint_template, iteration)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(checkpoint, iteration=iteration)
        gate_seed = int(args.gate_seed) + gate_index * 1_000_003
        stats = evaluate_hold(trainer, args, gate_seed=gate_seed)
        failures = check_gate(stats, args)
        passed = not failures
        pass_streak = pass_streak + 1 if passed else 0
        entry = {
            "iteration": iteration,
            "samples": iteration * sample_count,
            "checkpoint": str(checkpoint),
            "gate_seed": gate_seed,
            "stats": asdict(stats),
            "failures": failures,
            "pass_gate": passed,
            "consecutive_passes": pass_streak,
            "qualified": pass_streak >= int(args.required_consecutive_passes),
            "total_wall_seconds": time.perf_counter() - total_t0,
        }
        gate_history.append(entry)
        gate_index += 1
        return entry

    # A randomly initialized policy is a legitimate zero-training candidate.
    # Evaluating it also prevents a coarse chunk size from overstating time.
    entry = evaluate_iteration(0)
    if entry["qualified"]:
        first_pass = entry

    graph = None
    if first_pass is None:
        trainer.reserve_update_buffers(buffer)
        seed_counter = rl.make_seed_counter(int(args.seed), device=device)
        with wp.ScopedCapture(device=device) as capture:
            rl.collect_ppo_rollout_seed_counter(env, trainer, buffer, seed_counter=seed_counter)
            trainer.update(buffer, read_stats=False)
        graph = capture.graph

    while first_pass is None and completed_iterations < int(args.iterations):
        chunk = min(int(args.chunk_iterations), int(args.iterations) - completed_iterations)
        for _ in range(chunk):
            assert graph is not None
            wp.capture_launch(graph)
        completed_iterations += chunk
        trainer.iteration = completed_iterations
        entry = evaluate_iteration(completed_iterations)
        if entry["qualified"]:
            first_pass = entry

    result = {
        "engine": "phoenx_dr_legs_hold_warp_ppo_train_to_gate",
        "metric": "graph training plus seeded no-reset physical hold gate",
        "seed": int(args.seed),
        "gate_seed": int(args.gate_seed),
        "world_count": int(args.world_count),
        "rollout_steps": int(args.rollout_steps),
        "sim_substeps": int(args.sim_substeps),
        "solver_iterations": int(args.solver_iterations),
        "velocity_iterations": int(args.velocity_iterations),
        "required_consecutive_passes": int(args.required_consecutive_passes),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-seed", type=int, default=1000)
    parser.add_argument("--world-count", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--chunk-iterations", type=int, default=50)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--sim-substeps", type=int, default=20)
    parser.add_argument("--collision-refresh-interval", type=int, default=4)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=(512, 512, 512))
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--required-consecutive-passes", type=int, default=2)
    parser.add_argument("--eval-world-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--eval-kick-speed", type=float, default=0.15)
    parser.add_argument("--gate-max-fall-fraction", type=float, default=0.0)
    parser.add_argument("--gate-min-survival-fraction", type=float, default=0.98)
    parser.add_argument("--gate-min-success", type=float, default=0.75)
    parser.add_argument("--gate-min-pelvis-height", type=float, default=0.20)
    parser.add_argument("--gate-max-pelvis-height", type=float, default=0.34)
    parser.add_argument("--gate-min-upright-cos", type=float, default=0.85)
    parser.add_argument("--gate-max-horizontal-drift", type=float, default=0.12)
    parser.add_argument("--gate-max-anchor-residual", type=float, default=1.0e-3)
    return parser


def main() -> int:
    args = _make_parser().parse_args()
    result = benchmark_train_to_gate(args)
    return 0 if result["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
