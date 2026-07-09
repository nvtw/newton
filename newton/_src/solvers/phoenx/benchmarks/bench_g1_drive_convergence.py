# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure PhoenX G1 drive convergence across solver settings.

The benchmark applies a fixed nanoG1-v3-style leg action target to the full
coordinate G1 environment, then compares lower-cost PhoenX solver settings to a
higher-resolution PhoenX reference. It is intended as a small, repeatable study
for whether the RL recipe's solver settings are plausible for stiff Unitree PD
drives, not as a policy-quality benchmark. Contact support diagnostics
intentionally read back per substep, so reported timings are diagnostic-only.

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

import newton
import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.rl_training.g1 import g1_apply_actions_kernel, g1_increment_episode_steps_kernel
from newton._src.solvers.phoenx.rl_training.g1_diagnostics import (
    G1_FOOT_CONTACT_METRIC_ACTIVE_NORMAL_COUNT,
    G1_FOOT_CONTACT_METRIC_ACTIVE_TANGENT_COUNT,
    G1_FOOT_CONTACT_METRIC_COUNT,
    G1_FOOT_CONTACT_METRIC_COUNT_TOTAL,
    G1_FOOT_CONTACT_METRIC_FRICTION_LOAD,
    G1_FOOT_CONTACT_METRIC_FRICTION_LOAD_RATIO_SUM,
    G1_FOOT_CONTACT_METRIC_HIGH_TANGENT_RATIO_COUNT,
    G1_FOOT_CONTACT_METRIC_NORMAL_IMPULSE,
    G1_FOOT_CONTACT_METRIC_SPECULATIVE_COUNT,
    G1_FOOT_CONTACT_METRIC_TANGENT_BIAS,
    G1_FOOT_CONTACT_METRIC_TANGENT_IMPULSE,
    G1_FOOT_CONTACT_METRIC_TANGENT_NORMAL_RATIO_SUM,
    scan_g1_foot_contact_metrics,
)


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
    "work_5x8_v1": SolverSetting("work_5x8_v1", 5, 8, 1),
    "work_10x4_v1": SolverSetting("work_10x4_v1", 10, 4, 1),
    "work_20x2_v1": SolverSetting("work_20x2_v1", 20, 2, 1),
    "work_40x1_v1": SolverSetting("work_40x1_v1", 40, 1, 1),
    "phoenx_10x16_v1": SolverSetting("phoenx_10x16_v1", 10, 16, 1),
    "phoenx_10x32_v1": SolverSetting("phoenx_10x32_v1", 10, 32, 1),
    "phoenx_10x64_v1": SolverSetting("phoenx_10x64_v1", 10, 64, 1),
    "velocity_10x4_v0": SolverSetting("velocity_10x4_v0", 10, 4, 0),
    "velocity_10x4_v1": SolverSetting("velocity_10x4_v1", 10, 4, 1),
    "velocity_10x4_v2": SolverSetting("velocity_10x4_v2", 10, 4, 2),
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


def _parse_int_or_auto(value: str) -> int | str:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1 or auto")
    return parsed


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
        actuation_model=str(args.actuation_model),
        articulation_mode=str(args.articulation_mode),
        command=(0.0, 0.0, 0.0),
        max_episode_steps=0,
        auto_reset=False,
        joint_friction_model=str(args.joint_friction_model),
        joint_friction_scale=float(args.joint_friction_scale),
        parse_meshes=bool(args.parse_meshes),
        contact_geometry=str(args.contact_geometry),
        controlled_action_count=g1_recipe.CONTROLLED_ACTION_COUNT,
        rigid_contact_max_per_world=int(args.rigid_contact_max_per_world),
        threads_per_world=args.threads_per_world,
        multi_world_scheduler=str(args.multi_world_scheduler),
        prepare_refresh_stride=args.prepare_refresh_stride,
    )
    env = rl.EnvG1PhoenX(config, device=device)
    if float(args.armature_scale) != 1.0:
        env.model.joint_armature.assign(env.model.joint_armature.numpy() * np.float32(args.armature_scale))
        env.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    return env


# This mirrors EnvG1PhoenX.step(auto_reset=False) so support impulses can
# be accumulated across all physics substeps in one policy frame.
def _step_env_with_support_metrics(
    env: rl.EnvG1PhoenX,
    actions: wp.array2d[wp.float32],
    foot_metrics: wp.array3d[wp.float32],
) -> tuple[np.ndarray, ...]:
    wp.launch(
        g1_apply_actions_kernel,
        dim=(env.world_count, env.action_dim),
        inputs=[
            actions,
            env.default_joint_pos,
            env.ctrl_lower,
            env.ctrl_upper,
            env.config.action_scale,
            int(env.config.controlled_action_count),
            env.dof_stride,
            env.coord_stride,
            int(bool(env.model.use_coord_layout_targets)),
        ],
        outputs=[env.current_actions, env.control.joint_target_q],
        device=env.device,
    )

    substeps = int(env.config.sim_substeps)
    sub_dt = float(env.config.frame_dt) / float(substeps)
    contact_count_accum = np.zeros((env.world_count, 2), dtype=np.float64)
    normal_impulse_accum = np.zeros((env.world_count, 2), dtype=np.float64)
    tangent_impulse_accum = np.zeros((env.world_count, 2), dtype=np.float64)
    ratio_sum_accum = np.zeros((env.world_count, 2), dtype=np.float64)
    speculative_count_accum = np.zeros((env.world_count, 2), dtype=np.float64)
    tangent_bias_accum = np.zeros((env.world_count, 2), dtype=np.float64)
    high_tangent_ratio_count_accum = np.zeros((env.world_count, 2), dtype=np.float64)
    active_normal_count_accum = np.zeros((env.world_count, 2), dtype=np.float64)
    active_tangent_count_accum = np.zeros((env.world_count, 2), dtype=np.float64)
    friction_load_accum = np.zeros((env.world_count, 2), dtype=np.float64)
    friction_load_ratio_sum_accum = np.zeros((env.world_count, 2), dtype=np.float64)

    for substep in range(substeps):
        env.state_0.clear_forces()
        env.model.collide(env.state_0, env.contacts)
        if substep == substeps - 1:
            env._gather_actuator_force()
        env.solver.step(env.state_0, env.state_1, env.control, env.contacts, sub_dt)
        scan_g1_foot_contact_metrics(env, foot_metrics)
        metrics = foot_metrics.numpy()
        contact_count_accum += metrics[:, :, G1_FOOT_CONTACT_METRIC_COUNT]
        normal_impulse_accum += metrics[:, :, G1_FOOT_CONTACT_METRIC_NORMAL_IMPULSE]
        tangent_impulse_accum += metrics[:, :, G1_FOOT_CONTACT_METRIC_TANGENT_IMPULSE]
        ratio_sum_accum += metrics[:, :, G1_FOOT_CONTACT_METRIC_TANGENT_NORMAL_RATIO_SUM]
        speculative_count_accum += metrics[:, :, G1_FOOT_CONTACT_METRIC_SPECULATIVE_COUNT]
        tangent_bias_accum += metrics[:, :, G1_FOOT_CONTACT_METRIC_TANGENT_BIAS]
        high_tangent_ratio_count_accum += metrics[:, :, G1_FOOT_CONTACT_METRIC_HIGH_TANGENT_RATIO_COUNT]
        active_normal_count_accum += metrics[:, :, G1_FOOT_CONTACT_METRIC_ACTIVE_NORMAL_COUNT]
        active_tangent_count_accum += metrics[:, :, G1_FOOT_CONTACT_METRIC_ACTIVE_TANGENT_COUNT]
        friction_load_accum += metrics[:, :, G1_FOOT_CONTACT_METRIC_FRICTION_LOAD]
        friction_load_ratio_sum_accum += metrics[:, :, G1_FOOT_CONTACT_METRIC_FRICTION_LOAD_RATIO_SUM]
        env.state_0, env.state_1 = env.state_1, env.state_0

    wp.launch(g1_increment_episode_steps_kernel, dim=env.world_count, outputs=[env.episode_steps], device=env.device)
    env.observe()
    wp.copy(env.step_rewards, env.rewards)
    wp.copy(env.step_dones, env.dones)
    wp.copy(env.step_successes, env.successes)
    wp.copy(env.previous_actions, env.current_actions)

    ratio_mean = np.divide(
        ratio_sum_accum,
        contact_count_accum,
        out=np.zeros_like(ratio_sum_accum),
        where=contact_count_accum > 1.0e-12,
    )
    friction_load_ratio_mean = np.divide(
        friction_load_ratio_sum_accum,
        active_normal_count_accum,
        out=np.zeros_like(friction_load_ratio_sum_accum),
        where=active_normal_count_accum > 1.0e-12,
    )
    substeps_f = float(substeps)
    return (
        contact_count_accum / substeps_f,
        normal_impulse_accum,
        tangent_impulse_accum,
        ratio_mean,
        speculative_count_accum / substeps_f,
        tangent_bias_accum,
        high_tangent_ratio_count_accum / substeps_f,
        active_normal_count_accum / substeps_f,
        active_tangent_count_accum / substeps_f,
        friction_load_accum,
        friction_load_ratio_mean,
    )


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
    if args.initial_base_z is not None:
        q_init = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
        qd_init = env.state_0.joint_qd.numpy().reshape(env.world_count, env.dof_stride)
        q_init[:, 2] = np.float32(args.initial_base_z)
        qd_init.fill(0.0)
        env.state_0.joint_q.assign(q_init.reshape(-1))
        env.state_0.joint_qd.assign(qd_init.reshape(-1))
        newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)
        env.observe()

    step_count = int(args.steps)
    initial_q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
    initial_controlled_q = initial_q[:, 7 : 7 + g1_recipe.CONTROLLED_ACTION_COUNT].copy()
    foot_metrics = wp.zeros((env.world_count, 2, G1_FOOT_CONTACT_METRIC_COUNT_TOTAL), dtype=wp.float32, device=device)
    foot_counts = np.zeros((step_count, env.world_count, 2), dtype=np.float64)
    foot_normal_impulses = np.zeros((step_count, env.world_count, 2), dtype=np.float64)
    foot_tangent_impulses = np.zeros((step_count, env.world_count, 2), dtype=np.float64)
    foot_tangent_normal_ratios = np.zeros((step_count, env.world_count, 2), dtype=np.float64)
    foot_speculative_counts = np.zeros((step_count, env.world_count, 2), dtype=np.float64)
    foot_tangent_biases = np.zeros((step_count, env.world_count, 2), dtype=np.float64)
    foot_high_tangent_ratio_counts = np.zeros((step_count, env.world_count, 2), dtype=np.float64)
    foot_active_normal_counts = np.zeros((step_count, env.world_count, 2), dtype=np.float64)
    foot_active_tangent_counts = np.zeros((step_count, env.world_count, 2), dtype=np.float64)
    foot_friction_loads = np.zeros((step_count, env.world_count, 2), dtype=np.float64)
    foot_friction_load_ratios = np.zeros((step_count, env.world_count, 2), dtype=np.float64)

    t0 = time.perf_counter()
    for step in range(step_count):
        (
            counts,
            normal_impulses,
            tangent_impulses,
            tangent_normal_ratios,
            speculative_counts,
            tangent_biases,
            high_tangent_ratio_counts,
            active_normal_counts,
            active_tangent_counts,
            friction_loads,
            friction_load_ratios,
        ) = _step_env_with_support_metrics(env, actions, foot_metrics)
        foot_counts[step] = counts
        foot_normal_impulses[step] = normal_impulses
        foot_tangent_impulses[step] = tangent_impulses
        foot_tangent_normal_ratios[step] = tangent_normal_ratios
        foot_speculative_counts[step] = speculative_counts
        foot_tangent_biases[step] = tangent_biases
        foot_high_tangent_ratio_counts[step] = high_tangent_ratio_counts
        foot_active_normal_counts[step] = active_normal_counts
        foot_active_tangent_counts[step] = active_tangent_counts
        foot_friction_loads[step] = friction_loads
        foot_friction_load_ratios[step] = friction_load_ratios
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
    action_delta = np.float32(env.config.action_scale) * action_row[: g1_recipe.CONTROLLED_ACTION_COUNT]
    active = np.abs(action_delta) > np.float32(1.0e-7)
    tracking_ratio_mean = None
    tracking_ratio_std = None
    if np.any(active):
        tracking_ratio = (controlled_q[:, active] - initial_controlled_q[:, active]) / action_delta[active][None, :]
        tracking_ratio_mean = float(np.mean(tracking_ratio))
        tracking_ratio_std = float(np.std(tracking_ratio))
    upright_cos = -obs[:, 5]
    support_normal_impulse = np.sum(foot_normal_impulses, axis=2)
    support_tangent_impulse = np.sum(foot_tangent_impulses, axis=2)
    support_tangent_bias = np.sum(foot_tangent_biases, axis=2)
    support_friction_load = np.sum(foot_friction_loads, axis=2)
    foot_speculative_fraction = np.divide(
        foot_speculative_counts, foot_counts, out=np.zeros_like(foot_speculative_counts), where=foot_counts > 1.0e-12
    )
    foot_high_tangent_ratio_fraction = np.divide(
        foot_high_tangent_ratio_counts,
        foot_active_normal_counts,
        out=np.zeros_like(foot_high_tangent_ratio_counts),
        where=foot_active_normal_counts > 1.0e-12,
    )
    foot_active_normal_fraction = np.divide(
        foot_active_normal_counts,
        foot_counts,
        out=np.zeros_like(foot_active_normal_counts),
        where=foot_counts > 1.0e-12,
    )
    foot_active_tangent_fraction = np.divide(
        foot_active_tangent_counts,
        foot_active_normal_counts,
        out=np.zeros_like(foot_active_tangent_counts),
        where=foot_active_normal_counts > 1.0e-12,
    )

    return {
        "setting": setting.name,
        "sim_substeps": int(setting.sim_substeps),
        "solver_iterations": int(setting.solver_iterations),
        "velocity_iterations": int(setting.velocity_iterations),
        "physics_dt": float(env.config.frame_dt) / float(setting.sim_substeps),
        "elapsed_seconds": float(elapsed),
        "control_steps_per_s": float(env.world_count * step_count) / elapsed,
        "physics_steps_per_s": float(env.world_count * step_count * int(setting.sim_substeps)) / elapsed,
        "joint_target_rms_rad": float(np.sqrt(np.mean(target_err * target_err))),
        "joint_target_max_abs_rad": float(np.max(np.abs(target_err))),
        "joint_tracking_ratio_mean": tracking_ratio_mean,
        "joint_tracking_ratio_std": tracking_ratio_std,
        "foot_contact_count_mean": float(np.mean(foot_counts)),
        "foot_contact_count_min": float(np.min(foot_counts)),
        "left_foot_contact_count_mean": float(np.mean(foot_counts[:, :, 0])),
        "right_foot_contact_count_mean": float(np.mean(foot_counts[:, :, 1])),
        "support_normal_impulse_mean": float(np.mean(support_normal_impulse)),
        "support_normal_impulse_min": float(np.min(support_normal_impulse)),
        "support_tangent_impulse_mean": float(np.mean(support_tangent_impulse)),
        "foot_tangent_normal_ratio_mean": float(np.mean(foot_tangent_normal_ratios)),
        "foot_tangent_normal_ratio_max": float(np.max(foot_tangent_normal_ratios)),
        "foot_speculative_fraction_mean": float(np.mean(foot_speculative_fraction)),
        "foot_speculative_fraction_max": float(np.max(foot_speculative_fraction)),
        "support_tangent_bias_mean": float(np.mean(support_tangent_bias)),
        "foot_high_tangent_ratio_fraction_mean": float(np.mean(foot_high_tangent_ratio_fraction)),
        "foot_active_normal_fraction_mean": float(np.mean(foot_active_normal_fraction)),
        "foot_active_tangent_fraction_mean": float(np.mean(foot_active_tangent_fraction)),
        "support_friction_load_mean": float(np.mean(support_friction_load)),
        "foot_friction_load_ratio_mean": float(np.mean(foot_friction_load_ratios)),
        "base_height_mean_m": float(np.mean(q[:, 2])),
        "base_height_min_m": float(np.min(q[:, 2])),
        "upright_cos_mean": float(np.mean(upright_cos)),
        "upright_cos_min": float(np.min(upright_cos)),
        "fall_fraction": float(np.mean(dones > 0.5)),
        "q": q,
        "qd": qd,
        "foot_counts": foot_counts,
        "support_normal_impulse": support_normal_impulse,
        "support_tangent_impulse": support_tangent_impulse,
        "foot_tangent_normal_ratios": foot_tangent_normal_ratios,
        "support_tangent_bias": support_tangent_bias,
        "foot_speculative_fraction": foot_speculative_fraction,
        "foot_high_tangent_ratio_fraction": foot_high_tangent_ratio_fraction,
        "foot_active_normal_fraction": foot_active_normal_fraction,
        "foot_active_tangent_fraction": foot_active_tangent_fraction,
        "support_friction_load": support_friction_load,
        "foot_friction_load_ratios": foot_friction_load_ratios,
    }


def _attach_reference_errors(results: list[dict[str, Any]], reference: dict[str, Any]) -> None:
    ref_q = reference["q"]
    ref_qd = reference["qd"]
    ref_joint = ref_q[:, 7 : 7 + rl.ACTION_DIM_G1]
    ref_qd_joint = ref_qd[:, 6 : 6 + rl.ACTION_DIM_G1]
    ref_base = ref_q[:, :7]
    ref_foot_counts = reference["foot_counts"]
    ref_support_normal = reference["support_normal_impulse"]
    ref_support_tangent = reference["support_tangent_impulse"]
    ref_tangent_normal_ratios = reference["foot_tangent_normal_ratios"]
    ref_support_tangent_bias = reference["support_tangent_bias"]
    ref_speculative_fraction = reference["foot_speculative_fraction"]
    ref_high_tangent_ratio_fraction = reference["foot_high_tangent_ratio_fraction"]
    ref_active_normal_fraction = reference["foot_active_normal_fraction"]
    ref_active_tangent_fraction = reference["foot_active_tangent_fraction"]
    ref_support_friction_load = reference["support_friction_load"]
    ref_friction_load_ratios = reference["foot_friction_load_ratios"]
    ref_tracking_ratio = reference["joint_tracking_ratio_mean"]
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
        foot_count_err = result["foot_counts"] - ref_foot_counts
        normal_impulse_err = result["support_normal_impulse"] - ref_support_normal
        tangent_impulse_err = result["support_tangent_impulse"] - ref_support_tangent
        tangent_normal_ratio_err = result["foot_tangent_normal_ratios"] - ref_tangent_normal_ratios
        tangent_bias_err = result["support_tangent_bias"] - ref_support_tangent_bias
        speculative_fraction_err = result["foot_speculative_fraction"] - ref_speculative_fraction
        high_tangent_ratio_fraction_err = result["foot_high_tangent_ratio_fraction"] - ref_high_tangent_ratio_fraction
        active_normal_fraction_err = result["foot_active_normal_fraction"] - ref_active_normal_fraction
        active_tangent_fraction_err = result["foot_active_tangent_fraction"] - ref_active_tangent_fraction
        friction_load_err = result["support_friction_load"] - ref_support_friction_load
        friction_load_ratio_err = result["foot_friction_load_ratios"] - ref_friction_load_ratios
        result["base_ref_rms"] = float(np.sqrt(np.mean(base_err * base_err)))
        result["foot_contact_count_ref_rmse"] = float(np.sqrt(np.mean(foot_count_err * foot_count_err)))
        result["support_normal_impulse_ref_rmse"] = float(np.sqrt(np.mean(normal_impulse_err * normal_impulse_err)))
        result["support_normal_impulse_ref_delta_mean"] = float(np.mean(normal_impulse_err))
        result["support_tangent_impulse_ref_rmse"] = float(np.sqrt(np.mean(tangent_impulse_err * tangent_impulse_err)))
        result["support_tangent_impulse_ref_delta_mean"] = float(np.mean(tangent_impulse_err))
        result["foot_tangent_normal_ratio_ref_rmse"] = float(
            np.sqrt(np.mean(tangent_normal_ratio_err * tangent_normal_ratio_err))
        )
        result["foot_tangent_normal_ratio_ref_delta_mean"] = float(np.mean(tangent_normal_ratio_err))
        result["support_tangent_bias_ref_rmse"] = float(np.sqrt(np.mean(tangent_bias_err * tangent_bias_err)))
        result["support_tangent_bias_ref_delta_mean"] = float(np.mean(tangent_bias_err))
        result["foot_speculative_fraction_ref_delta_mean"] = float(np.mean(speculative_fraction_err))
        result["foot_high_tangent_ratio_fraction_ref_delta_mean"] = float(np.mean(high_tangent_ratio_fraction_err))
        result["foot_active_normal_fraction_ref_delta_mean"] = float(np.mean(active_normal_fraction_err))
        result["foot_active_tangent_fraction_ref_delta_mean"] = float(np.mean(active_tangent_fraction_err))
        result["support_friction_load_ref_delta_mean"] = float(np.mean(friction_load_err))
        result["support_friction_load_ref_rmse"] = float(np.sqrt(np.mean(friction_load_err * friction_load_err)))
        result["foot_friction_load_ratio_ref_delta_mean"] = float(np.mean(friction_load_ratio_err))
        if ref_tracking_ratio is not None and result["joint_tracking_ratio_mean"] is not None:
            result["joint_tracking_ratio_ref_delta"] = float(result["joint_tracking_ratio_mean"] - ref_tracking_ratio)
        else:
            result["joint_tracking_ratio_ref_delta"] = None
        del result["q"]
        del result["qd"]
        del result["foot_counts"]
        del result["support_normal_impulse"]
        del result["support_tangent_impulse"]
        del result["foot_tangent_normal_ratios"]
        del result["support_tangent_bias"]
        del result["foot_speculative_fraction"]
        del result["foot_high_tangent_ratio_fraction"]
        del result["foot_active_normal_fraction"]
        del result["foot_active_tangent_fraction"]
        del result["support_friction_load"]
        del result["foot_friction_load_ratios"]


def benchmark_g1_drive_convergence(args: argparse.Namespace) -> dict[str, Any]:
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("G1 drive convergence benchmark requires a CUDA device")
    if int(args.steps) <= 0:
        raise ValueError("steps must be positive")
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
        "metric": "fixed G1 leg-target response, substep-averaged contact support, and frame impulses versus high-resolution PhoenX reference",
        "device": device.name,
        "timing_includes_metric_readback": True,
        "world_count": int(args.world_count),
        "steps": int(args.steps),
        "policy_frame_dt": float(g1_recipe.FRAME_DT),
        "action_pattern": str(args.action_pattern),
        "action_amplitude": float(args.action_amplitude),
        "controlled_action_count": int(g1_recipe.CONTROLLED_ACTION_COUNT),
        "contact_geometry": str(args.contact_geometry),
        "joint_friction_model": str(args.joint_friction_model),
        "joint_friction_scale": float(args.joint_friction_scale),
        "actuation_model": str(args.actuation_model),
        "articulation_mode": str(args.articulation_mode),
        "armature_scale": float(args.armature_scale),
        "initial_base_z": None if args.initial_base_z is None else float(args.initial_base_z),
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
    parser.add_argument("--initial-base-z", type=float, default=None)
    parser.add_argument("--joint-friction-model", choices=("hard", "mujoco"), default=g1_recipe.JOINT_FRICTION_MODEL)
    parser.add_argument("--joint-friction-scale", type=float, default=g1_recipe.JOINT_FRICTION_SCALE)
    parser.add_argument(
        "--actuation-model", choices=("explicit_torque", "constraint_drive"), default="constraint_drive"
    )
    parser.add_argument(
        "--articulation-mode", choices=("maximal", "maximal_projected", "hybrid", "reduced"), default="maximal"
    )
    parser.add_argument("--armature-scale", type=float, default=1.0)
    parser.add_argument("--parse-meshes", action="store_true")
    parser.add_argument("--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=g1_recipe.CONTACT_GEOMETRY)
    parser.add_argument("--rigid-contact-max-per-world", type=int, default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
    parser.add_argument("--threads-per-world", default=g1_recipe.THREADS_PER_WORLD)
    parser.add_argument("--multi-world-scheduler", default=g1_recipe.MULTI_WORLD_SCHEDULER)
    parser.add_argument("--prepare-refresh-stride", type=_parse_int_or_auto, default=g1_recipe.PREPARE_REFRESH_STRIDE)
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
