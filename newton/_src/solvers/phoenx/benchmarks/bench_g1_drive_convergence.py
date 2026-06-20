# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure PhoenX G1 drive convergence across solver settings.

The benchmark applies a fixed nanoG1-v3-style leg action target to the full
coordinate G1 environment, then compares lower-cost PhoenX solver settings to a
higher-resolution PhoenX reference. It is intended as a small, repeatable study
for whether the RL recipe's solver settings are plausible for stiff Unitree PD
drives, not as a policy-quality benchmark.

Examples:
    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_drive_convergence \
        --steps 20 --world-count 4 --json-indent 2
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe


@dataclass(frozen=True)
class SolverSetting:
    """PhoenX solver setting used by the drive convergence sweep."""

    name: str
    sim_substeps: int
    solver_iterations: int
    velocity_iterations: int


_SETTINGS: dict[str, SolverSetting] = {
    "rl_current": SolverSetting("rl_current", 5, 2, 1),
    "phoenx_5x4": SolverSetting("phoenx_5x4", 5, 4, 1),
    "phoenx_5x8": SolverSetting("phoenx_5x8", 5, 8, 2),
    "phoenx_10x4": SolverSetting("phoenx_10x4", 10, 4, 1),
    "phoenx_10x8": SolverSetting("phoenx_10x8", 10, 8, 2),
    "phoenx_20x8": SolverSetting("phoenx_20x8", 20, 8, 2),
}


def _parse_csv(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("expected at least one setting name")
    unknown = sorted(set(names) - set(_SETTINGS))
    if unknown:
        known = ", ".join(sorted(_SETTINGS))
        raise argparse.ArgumentTypeError(f"unknown settings {unknown}; known settings: {known}")
    return names


def _leg_action_pattern(name: str, amplitude: float) -> np.ndarray:
    actions = np.zeros(rl.ACTION_DIM_G1, dtype=np.float32)
    if name == "zero":
        return actions
    if name == "leg_step":
        leg = np.asarray((0.35, -0.25, 0.20, 0.65, -0.55, 0.25), dtype=np.float32)
        mirrored = np.asarray((0.35, 0.25, -0.20, 0.65, -0.55, -0.25), dtype=np.float32)
        pattern = np.concatenate((leg, mirrored))
    elif name == "leg_symmetric":
        pattern = np.asarray((0.30, 0.0, 0.0, 0.55, -0.45, 0.0) * 2, dtype=np.float32)
    else:
        raise ValueError(f"unknown action pattern: {name}")
    max_abs = float(np.max(np.abs(pattern)))
    if max_abs > 0.0:
        pattern = pattern / np.float32(max_abs)
    actions[:12] = np.float32(amplitude) * pattern
    return actions


def _make_env(setting: SolverSetting, args: argparse.Namespace, *, device: wp.context.Device) -> rl.EnvG1PhoenX:
    config = rl.ConfigEnvG1PhoenX(
        world_count=int(args.world_count),
        sim_substeps=int(setting.sim_substeps),
        solver_iterations=int(setting.solver_iterations),
        velocity_iterations=int(setting.velocity_iterations),
        command=(0.0, 0.0, 0.0),
        max_episode_steps=0,
        auto_reset=False,
        parse_meshes=bool(args.parse_meshes),
        controlled_action_count=g1_recipe.CONTROLLED_ACTION_COUNT,
        rigid_contact_max_per_world=int(args.rigid_contact_max_per_world),
        threads_per_world=args.threads_per_world,
        multi_world_scheduler=str(args.multi_world_scheduler),
        prepare_refresh_stride=args.prepare_refresh_stride,
    )
    return rl.EnvG1PhoenX(config, device=device)


def _run_setting(
    setting: SolverSetting,
    args: argparse.Namespace,
    action_row: np.ndarray,
    *,
    device: wp.context.Device,
) -> dict[str, Any]:
    env = _make_env(setting, args, device=device)
    actions_np = np.tile(action_row.astype(np.float32, copy=False), (env.world_count, 1))
    actions = wp.array(actions_np, dtype=wp.float32, device=device)

    env.reset()
    t0 = time.perf_counter()
    for _ in range(int(args.steps)):
        env.step(actions)
    wp.synchronize_device(device)
    elapsed = max(time.perf_counter() - t0, 1.0e-12)

    obs = env.obs.numpy()
    q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
    qd = env.state_0.joint_qd.numpy().reshape(env.world_count, env.dof_stride)
    dones = env.dones.numpy()
    target = np.asarray(env.default_joint_pos.numpy(), dtype=np.float32).copy()
    target[: g1_recipe.CONTROLLED_ACTION_COUNT] += (
        np.float32(env.config.action_scale) * action_row[: g1_recipe.CONTROLLED_ACTION_COUNT]
    )
    joint_q = q[:, 7 : 7 + env.action_dim]
    controlled_q = joint_q[:, : g1_recipe.CONTROLLED_ACTION_COUNT]
    controlled_target = target[: g1_recipe.CONTROLLED_ACTION_COUNT]
    target_err = controlled_q - controlled_target[None, :]
    upright_cos = -obs[:, 5]

    return {
        "setting": setting.name,
        "sim_substeps": int(setting.sim_substeps),
        "solver_iterations": int(setting.solver_iterations),
        "velocity_iterations": int(setting.velocity_iterations),
        "physics_dt": float(env.config.frame_dt) / float(setting.sim_substeps),
        "elapsed_seconds": float(elapsed),
        "control_steps_per_s": float(env.world_count * int(args.steps)) / elapsed,
        "physics_steps_per_s": float(env.world_count * int(args.steps) * int(setting.sim_substeps)) / elapsed,
        "joint_target_rms_rad": float(np.sqrt(np.mean(target_err * target_err))),
        "joint_target_max_abs_rad": float(np.max(np.abs(target_err))),
        "base_height_mean_m": float(np.mean(q[:, 2])),
        "base_height_min_m": float(np.min(q[:, 2])),
        "upright_cos_mean": float(np.mean(upright_cos)),
        "upright_cos_min": float(np.min(upright_cos)),
        "fall_fraction": float(np.mean(dones > 0.5)),
        "q": q,
        "qd": qd,
    }


def _attach_reference_errors(results: list[dict[str, Any]], reference: dict[str, Any]) -> None:
    ref_q = reference["q"]
    ref_qd = reference["qd"]
    ref_joint = ref_q[:, 7 : 7 + rl.ACTION_DIM_G1]
    ref_qd_joint = ref_qd[:, 6 : 6 + rl.ACTION_DIM_G1]
    ref_base = ref_q[:, :7]
    for result in results:
        q = result["q"]
        qd = result["qd"]
        joint = q[:, 7 : 7 + rl.ACTION_DIM_G1]
        qd_joint = qd[:, 6 : 6 + rl.ACTION_DIM_G1]
        base = q[:, :7]
        joint_err = joint - ref_joint
        qd_err = qd_joint - ref_qd_joint
        base_err = base - ref_base
        result["joint_q_ref_rms_rad"] = float(np.sqrt(np.mean(joint_err * joint_err)))
        result["joint_q_ref_max_abs_rad"] = float(np.max(np.abs(joint_err)))
        result["joint_qd_ref_rms_rad_s"] = float(np.sqrt(np.mean(qd_err * qd_err)))
        result["base_ref_rms"] = float(np.sqrt(np.mean(base_err * base_err)))
        del result["q"]
        del result["qd"]


def benchmark_g1_drive_convergence(args: argparse.Namespace) -> dict[str, Any]:
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("G1 drive convergence benchmark requires a CUDA device")
    action_row = _leg_action_pattern(str(args.action_pattern), float(args.action_amplitude))
    setting_names = tuple(args.settings)
    reference_name = str(args.reference_setting)
    if reference_name not in _SETTINGS:
        raise ValueError(f"unknown reference setting: {reference_name}")
    if reference_name not in setting_names:
        setting_names = (*setting_names, reference_name)

    results = []
    for name in setting_names:
        results.append(_run_setting(_SETTINGS[name], args, action_row, device=device))

    reference = next(result for result in results if result["setting"] == reference_name)
    _attach_reference_errors(results, reference)
    return {
        "engine": "phoenx_g1_drive_convergence",
        "metric": "fixed G1 leg-target response versus high-resolution PhoenX reference",
        "device": device.name,
        "world_count": int(args.world_count),
        "steps": int(args.steps),
        "policy_frame_dt": float(g1_recipe.FRAME_DT),
        "action_pattern": str(args.action_pattern),
        "action_amplitude": float(args.action_amplitude),
        "controlled_action_count": int(g1_recipe.CONTROLLED_ACTION_COUNT),
        "reference_setting": reference_name,
        "nanog1_production_reference": {
            "source": "/home/twidmer/Documents/git/nanoG1/recipe.py TASK_FLAGS and web/g1_demo.c",
            "policy_frame_dt": 0.02,
            "physics_dt": 0.004,
            "decimation": 5,
            "newton_iterations": 2,
            "line_search_iterations": 3,
            "action_scale": 0.25,
            "controlled_action_count": 12,
            "leg_kp": (100.0, 100.0, 100.0, 150.0, 40.0, 40.0),
            "leg_unitree_kd": (2.0, 2.0, 2.0, 4.0, 2.0, 2.0),
            "leg_passive_damping": (2.0, 2.0, 2.0, 2.0, 1.0, 0.2),
            "leg_total_zero_velocity_damping": (4.0, 4.0, 4.0, 6.0, 3.0, 2.2),
        },
        "results": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--action-pattern", choices=("zero", "leg_step", "leg_symmetric"), default="leg_step")
    parser.add_argument("--action-amplitude", type=float, default=0.2)
    parser.add_argument(
        "--settings",
        type=_parse_csv,
        default=("rl_current", "phoenx_5x4", "phoenx_10x4", "phoenx_10x8"),
        help="Comma-separated setting names.",
    )
    parser.add_argument("--reference-setting", choices=tuple(sorted(_SETTINGS)), default="phoenx_20x8")
    parser.add_argument("--parse-meshes", action="store_true")
    parser.add_argument("--rigid-contact-max-per-world", type=int, default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
    parser.add_argument("--threads-per-world", default=g1_recipe.THREADS_PER_WORLD)
    parser.add_argument("--multi-world-scheduler", default=g1_recipe.MULTI_WORLD_SCHEDULER)
    parser.add_argument("--prepare-refresh-stride", default=g1_recipe.PREPARE_REFRESH_STRIDE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = benchmark_g1_drive_convergence(args)
    print(json.dumps(result, indent=args.json_indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
