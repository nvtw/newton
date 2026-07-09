# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure G1 exploration stability before expensive PPO training.

The benchmark replays identical smooth random leg actions from identical noisy
resets. It records first termination and joint closure only while a world is
still valid. Simulation is CUDA-graph captured; host readback is intentional
because this is a quality diagnostic, not a throughput benchmark.

Examples:
    uv run --extra dev -m \
        newton._src.solvers.phoenx.benchmarks.bench_g1_exploration_stability \
        --settings maximal_10x8_v1,reduced_5x2_v1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe


@dataclass(frozen=True)
class Setting:
    """Solver schedule exercised by the diagnostic."""

    name: str
    articulation_mode: str
    sim_substeps: int
    solver_iterations: int
    velocity_iterations: int


SETTINGS = {
    setting.name: setting
    for setting in (
        Setting("maximal_10x8_v1", "maximal", 10, 8, 1),
        Setting("maximal_20x4_v1", "maximal", 20, 4, 1),
        Setting("maximal_40x2_v1", "maximal", 40, 2, 1),
        Setting("maximal_projected_5x2_v1", "maximal_projected", 5, 2, 1),
        Setting("maximal_articulated_5x2_v1", "maximal_articulated", 5, 2, 1),
        Setting("maximal_articulated_4x2_v1", "maximal_articulated", 4, 2, 1),
        Setting("maximal_articulated_3x2_v1", "maximal_articulated", 3, 2, 1),
        Setting("maximal_articulated_5x1_v1", "maximal_articulated", 5, 1, 1),
        Setting("maximal_projected_5x1_v1", "maximal_projected", 5, 1, 1),
        Setting("maximal_projected_4x2_v1", "maximal_projected", 4, 2, 1),
        Setting("maximal_projected_3x2_v1", "maximal_projected", 3, 2, 1),
        Setting("maximal_projected_4x1_v1", "maximal_projected", 4, 1, 1),
        Setting("maximal_projected_3x1_v1", "maximal_projected", 3, 1, 1),
        Setting("hybrid_5x2_v1", "hybrid", 5, 2, 1),
        Setting("reduced_5x2_v1", "reduced", 5, 2, 1),
    )
}


def _parse_settings(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = sorted(set(names) - set(SETTINGS))
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"unknown settings {unknown}; choices are {sorted(SETTINGS)}")
    return names


def _action_frames(*, steps: int, worlds: int, seed: int, noise_std: float, smoothing: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = np.empty((steps, worlds, rl.ACTION_DIM_G1), dtype=np.float32)
    action = np.zeros((worlds, rl.ACTION_DIM_G1), dtype=np.float32)
    for step in range(steps):
        noise = rng.normal(0.0, noise_std, action.shape).astype(np.float32)
        action = np.clip(np.float32(smoothing) * action + np.float32(1.0 - smoothing) * noise, -1.0, 1.0)
        action[:, g1_recipe.CONTROLLED_ACTION_COUNT :] = 0.0
        frames[step] = action
    return frames


def _run_setting(
    setting: Setting,
    frames: np.ndarray,
    *,
    reset_seed: int,
    device: wp.context.Device,
) -> dict[str, object]:
    steps, worlds, _ = frames.shape
    env = rl.EnvG1PhoenX(
        g1_recipe.default_g1_env_config(
            world_count=worlds,
            articulation_mode=setting.articulation_mode,
            actuation_model="explicit_torque",
            sim_substeps=setting.sim_substeps,
            solver_iterations=setting.solver_iterations,
            velocity_iterations=setting.velocity_iterations,
            max_episode_steps=0,
            auto_reset=False,
            randomize_commands_on_reset=False,
        ),
        device=device,
    )
    env.reset_noisy(seed=reset_seed)
    actions = wp.zeros((worlds, rl.ACTION_DIM_G1), dtype=wp.float32, device=device)
    errors = wp.zeros(env.solver.world.num_constraints, dtype=wp.spatial_vector, device=device)

    with wp.ScopedCapture(device=device) as capture:
        env.step(actions)

    first_done = np.full(worlds, -1, dtype=np.int32)
    active_anchor_error_max = 0.0
    active_anchor_error_sq_sum = 0.0
    active_anchor_error_count = 0
    active_fraction = np.empty(steps, dtype=np.float64)
    start = time.perf_counter()
    for step in range(steps):
        actions.assign(frames[step])
        wp.capture_launch(capture.graph)
        done = env.dones.numpy() > 0.5
        env.solver.world.gather_constraint_errors(errors)
        joint_errors = errors.numpy()[: env.solver.world.num_joints].reshape(worlds, -1, 6)
        active = (first_done < 0) & ~done
        if np.any(active):
            anchor_error = np.linalg.norm(joint_errors[active, :, 3:5], axis=2)
            active_anchor_error_max = max(active_anchor_error_max, float(np.max(anchor_error)))
            active_anchor_error_sq_sum += float(np.sum(anchor_error * anchor_error))
            active_anchor_error_count += int(anchor_error.size)
        first_done[(first_done < 0) & done] = step + 1
        active_fraction[step] = float(np.mean(first_done < 0))
    elapsed = time.perf_counter() - start

    terminated = first_done >= 0
    return {
        **asdict(setting),
        "world_count": worlds,
        "steps": steps,
        "ever_terminated_fraction": float(np.mean(terminated)),
        "median_first_termination_step": (float(np.median(first_done[terminated])) if np.any(terminated) else None),
        "p10_first_termination_step": (
            float(np.percentile(first_done[terminated], 10.0)) if np.any(terminated) else None
        ),
        "active_fraction_final": float(active_fraction[-1]),
        "active_anchor_error_max": active_anchor_error_max,
        "active_anchor_error_rms": (
            float(np.sqrt(active_anchor_error_sq_sum / active_anchor_error_count)) if active_anchor_error_count else 0.0
        ),
        "diagnostic_seconds": elapsed,
        "control_steps_per_second_including_readback": float(worlds * steps / max(elapsed, 1.0e-12)),
    }


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("G1 exploration stability benchmark requires CUDA graph capture")
    if args.steps < 1 or args.world_count < 1:
        raise ValueError("steps and world-count must be positive")
    frames = _action_frames(
        steps=args.steps,
        worlds=args.world_count,
        seed=args.action_seed,
        noise_std=args.action_noise_std,
        smoothing=args.action_smoothing,
    )
    return {
        "engine": "phoenx_g1_exploration_stability",
        "metric": "matched smooth exploration, valid-world termination and joint closure",
        "device": device.name,
        "action_seed": args.action_seed,
        "reset_seed": args.reset_seed,
        "action_noise_std": args.action_noise_std,
        "action_smoothing": args.action_smoothing,
        "results": [
            _run_setting(SETTINGS[name], frames, reset_seed=args.reset_seed, device=device) for name in args.settings
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=_parse_settings, default=("maximal_10x8_v1", "reduced_5x2_v1"))
    parser.add_argument("--world-count", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--action-seed", type=int, default=1234)
    parser.add_argument("--reset-seed", type=int, default=77)
    parser.add_argument("--action-noise-std", type=float, default=0.45)
    parser.add_argument("--action-smoothing", type=float, default=0.92)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(json.dumps(benchmark(args), indent=args.json_indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
