# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare PhoenX G1 open-loop trajectories against nanoG1 host physics.

This benchmark compiles a tiny temporary C runner around nanoG1's host stepper
(`web/g1_host.c`) and compares state trajectories for the same reset pose,
action target pattern, v3 leg mask, action scale, and Unitree PD mode. The goal
is trajectory-level evidence for simulator parity before spending more time on
RL sample-efficiency tuning. Use ``--initial-base-z`` to lift the robot for
contact-free actuator/drive response checks.

Examples:
    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_open_loop_parity \
        --steps 20 --action-pattern zero --json-indent 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.rl as rl
from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.rl_training.g1_diagnostics import (
    G1_FOOT_CONTACT_METRIC_COUNT,
    G1_FOOT_CONTACT_METRIC_COUNT_TOTAL,
    G1_FOOT_CONTACT_METRIC_NORMAL_IMPULSE,
    G1_FOOT_CONTACT_METRIC_TANGENT_IMPULSE,
    G1_FOOT_CONTACT_METRIC_TANGENT_NORMAL_RATIO_SUM,
    scan_g1_foot_contact_metrics,
)

_NANOG1_WEB_DIR = Path("/home/twidmer/Documents/git/nanoG1/web")
_NANOG1_HOST = _NANOG1_WEB_DIR / "g1_host.c"


@dataclass(frozen=True)
class PhoenXSetting:
    """PhoenX setting compared against nanoG1 host physics."""

    name: str
    sim_substeps: int
    solver_iterations: int
    velocity_iterations: int


_SETTINGS: dict[str, PhoenXSetting] = {
    "fast_5x2": PhoenXSetting("fast_5x2", 5, 2, 1),
    "phoenx_5x4": PhoenXSetting("phoenx_5x4", 5, 4, 1),
    "recipe_default": PhoenXSetting(
        "recipe_default",
        g1_recipe.SIM_SUBSTEPS,
        g1_recipe.SOLVER_ITERATIONS,
        g1_recipe.VELOCITY_ITERATIONS,
    ),
    "phoenx_10x8": PhoenXSetting("phoenx_10x8", 10, 8, 2),
}


def _parse_csv(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("expected at least one setting")
    unknown = sorted(set(names) - set(_SETTINGS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown settings {unknown}; known settings: {sorted(_SETTINGS)}")
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


def _c_string_literal(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _host_runner_source(host_path: Path) -> str:
    host = _c_string_literal(host_path)
    return textwrap.dedent(
        f"""
        #include <math.h>
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>

        #include "{host}"

        static void world_to_base_host(const double q[4], const double v[3], double out[3]) {{
            double qinv[4] = {{q[0], -q[1], -q[2], -q[3]}};
            rot_vec_quat(v, qinv, out);
        }}

        static double upright_cos_host(const double* qpos) {{
            double g[3] = {{0.0, 0.0, -1.0}};
            double gb[3];
            world_to_base_host(qpos + 3, g, gb);
            return -gb[2];
        }}

        static void fill_actions(const char* pattern, double amplitude, double action[HC_NU]) {{
            for (int i = 0; i < HC_NU; ++i) action[i] = 0.0;
            if (strcmp(pattern, "zero") == 0) return;
            double values[12];
            if (strcmp(pattern, "leg_step") == 0) {{
                const double raw[12] = {{0.35, -0.25, 0.20, 0.65, -0.55, 0.25,
                                         0.35, 0.25, -0.20, 0.65, -0.55, -0.25}};
                for (int i = 0; i < 12; ++i) values[i] = raw[i] / 0.65;
            }} else if (strcmp(pattern, "leg_symmetric") == 0) {{
                const double raw[12] = {{0.30, 0.0, 0.0, 0.55, -0.45, 0.0,
                                         0.30, 0.0, 0.0, 0.55, -0.45, 0.0}};
                for (int i = 0; i < 12; ++i) values[i] = raw[i] / 0.55;
            }} else {{
                fprintf(stderr, "unknown action pattern: %s\\n", pattern);
                exit(2);
            }}
            for (int i = 0; i < 12; ++i) action[i] = amplitude * values[i];
        }}

        static void print_state_array(const double* data, int rows, int cols) {{
            putchar('[');
            for (int r = 0; r < rows; ++r) {{
                if (r) putchar(',');
                putchar('[');
                for (int c = 0; c < cols; ++c) {{
                    if (c) putchar(',');
                    printf("%.17g", data[r * cols + c]);
                }}
                putchar(']');
            }}
            putchar(']');
        }}

        static void print_scalar_array(const double* data, int count) {{
            putchar('[');
            for (int i = 0; i < count; ++i) {{
                if (i) putchar(',');
                printf("%.17g", data[i]);
            }}
            putchar(']');
        }}

        static int count_geom_contacts(int geom) {{
            int count = 0;
            for (int i = 0; i < ncon; ++i) {{
                if (con_g1[i] == geom || con_g2[i] == geom) count++;
            }}
            return count;
        }}

        static double contact_mu(int pair, int tangent_axis) {{
            double fri5[5] = {{hc_pair_friction[pair * 5 + 0], hc_pair_friction[pair * 5 + 0],
                              hc_pair_friction[pair * 5 + 1], hc_pair_friction[pair * 5 + 2],
                              hc_pair_friction[pair * 5 + 2]}};
            return fri5[tangent_axis - 1];
        }}

        static void contact_force_summary(int geom, double* normal_force, double* tangent_force) {{
            *normal_force = 0.0;
            *tangent_force = 0.0;
            for (int ci = 0; ci < ncon; ++ci) {{
                if (con_g1[ci] != geom && con_g2[ci] != geom) continue;
                int dim = phx_trace_con_dim[ci];
                int st = phx_trace_con_start[ci];
                if (dim <= 0 || st < 0 || st >= efc_nefc) continue;
                if (dim == 1) {{
                    double fn = phx_trace_efc_force[st];
                    if (fn > 0.0) *normal_force += fn;
                    continue;
                }}
                double fn = 0.0;
                double tangent_sq = 0.0;
                int pair = con_ipair[ci];
                for (int k = 1; k < dim; ++k) {{
                    int rp = st + 2 * (k - 1);
                    int rm = rp + 1;
                    if (rm >= efc_nefc) break;
                    double fp = phx_trace_efc_force[rp];
                    double fm = phx_trace_efc_force[rm];
                    if (fp < 0.0) fp = 0.0;
                    if (fm < 0.0) fm = 0.0;
                    double mu = contact_mu(pair, k);
                    fn += fp + fm;
                    double ft = mu * (fp - fm);
                    tangent_sq += ft * ft;
                }}
                *normal_force += fn;
                *tangent_force += sqrt(tangent_sq);
            }}
        }}

        static double qfrc_constraint_norm(void) {{
            double sum = 0.0;
            for (int i = 0; i < HC_NV; ++i) sum += qfrc_constraint[i] * qfrc_constraint[i];
            return sqrt(sum);
        }}

        int main(int argc, char** argv) {{
            if (argc < 10) {{
                fprintf(stderr, "usage: %s steps pattern amplitude decim dt newton ls stepper initial_base_z\\n", argv[0]);
                return 2;
            }}
            int steps = atoi(argv[1]);
            const char* pattern = argv[2];
            double amplitude = atof(argv[3]);
            int decim = atoi(argv[4]);
            double dt = atof(argv[5]);
            int newton = atoi(argv[6]);
            int ls = atoi(argv[7]);
            const char* stepper = argv[8];
            double initial_base_z = atof(argv[9]);
            if (steps <= 0 || decim <= 0 || dt <= 0.0 || newton <= 0 || ls <= 0) return 2;
            if (strcmp(stepper, "full") != 0 && strcmp(stepper, "smooth") != 0) return 2;

            hc_pd_unitree = 1;
            g_ls_iter = ls;

            double qpos[HC_NQ], qvel[HC_NV], qpn[HC_NQ], qvn[HC_NV], warmstart[HC_NV];
            double actions[HC_NU], ctrl[HC_NU];
            double* qpos_traj = (double*)calloc((size_t)(steps + 1) * HC_NQ, sizeof(double));
            double* qvel_traj = (double*)calloc((size_t)(steps + 1) * HC_NV, sizeof(double));
            double* base_z = (double*)calloc((size_t)(steps + 1), sizeof(double));
            double* upright = (double*)calloc((size_t)(steps + 1), sizeof(double));
            double* left_contacts = (double*)calloc((size_t)steps, sizeof(double));
            double* right_contacts = (double*)calloc((size_t)steps, sizeof(double));
            double* left_normal_force = (double*)calloc((size_t)steps, sizeof(double));
            double* right_normal_force = (double*)calloc((size_t)steps, sizeof(double));
            double* left_tangent_force = (double*)calloc((size_t)steps, sizeof(double));
            double* right_tangent_force = (double*)calloc((size_t)steps, sizeof(double));
            double* qfrc_norm = (double*)calloc((size_t)steps, sizeof(double));
            if (!qpos_traj || !qvel_traj || !base_z || !upright || !left_contacts || !right_contacts ||
                !left_normal_force || !right_normal_force || !left_tangent_force || !right_tangent_force || !qfrc_norm) return 3;

            memcpy(qpos, hc_key_qpos, sizeof(qpos));
            if (isfinite(initial_base_z)) qpos[2] = initial_base_z;
            memset(qvel, 0, sizeof(qvel));
            memset(warmstart, 0, sizeof(warmstart));
            fill_actions(pattern, amplitude, actions);

            memcpy(qpos_traj, qpos, sizeof(qpos));
            memcpy(qvel_traj, qvel, sizeof(qvel));
            base_z[0] = qpos[2];
            upright[0] = upright_cos_host(qpos);

            for (int t = 0; t < steps; ++t) {{
                for (int a = 0; a < HC_NU; ++a) {{
                    double c = actions[a];
                    if (c < -1.0) c = -1.0;
                    if (c > 1.0) c = 1.0;
                    if (a >= 12) c = 0.0;
                    double target = hc_key_qpos[7 + a] + 0.25 * c;
                    double lo = hc_act_ctrlrange[2 * a];
                    double hi = hc_act_ctrlrange[2 * a + 1];
                    if (target < lo) target = lo;
                    if (target > hi) target = hi;
                    ctrl[a] = target;
                }}
                for (int k = 0; k < decim; ++k) {{
                    if (strcmp(stepper, "smooth") == 0) {{
                        g1_smooth_step(qpos, qvel, ctrl, dt, qpn, qvn);
                        memset(qfrc_constraint, 0, sizeof(qfrc_constraint));
                        ncon = 0;
                    }} else {{
                        g1_full_step(qpos, qvel, ctrl, warmstart, dt, newton, qpn, qvn);
                        memcpy(warmstart, qacc_out, sizeof(warmstart));
                    }}
                    memcpy(qpos, qpn, sizeof(qpos));
                    memcpy(qvel, qvn, sizeof(qvel));
                }}
                memcpy(qpos_traj + (size_t)(t + 1) * HC_NQ, qpos, sizeof(qpos));
                memcpy(qvel_traj + (size_t)(t + 1) * HC_NV, qvel, sizeof(qvel));
                base_z[t + 1] = qpos[2];
                upright[t + 1] = upright_cos_host(qpos);
                left_contacts[t] = (double)count_geom_contacts(17);
                right_contacts[t] = (double)count_geom_contacts(31);
                contact_force_summary(17, &left_normal_force[t], &left_tangent_force[t]);
                contact_force_summary(31, &right_normal_force[t], &right_tangent_force[t]);
                qfrc_norm[t] = qfrc_constraint_norm();
            }}

            printf("{{\\"qpos\\":");
            print_state_array(qpos_traj, steps + 1, HC_NQ);
            printf(",\\"qvel\\":");
            print_state_array(qvel_traj, steps + 1, HC_NV);
            printf(",\\"base_z\\":");
            print_scalar_array(base_z, steps + 1);
            printf(",\\"upright_cos\\":");
            print_scalar_array(upright, steps + 1);
            printf(",\\"left_contacts\\":");
            print_scalar_array(left_contacts, steps);
            printf(",\\"right_contacts\\":");
            print_scalar_array(right_contacts, steps);
            printf(",\\"left_normal_force\\":");
            print_scalar_array(left_normal_force, steps);
            printf(",\\"right_normal_force\\":");
            print_scalar_array(right_normal_force, steps);
            printf(",\\"left_tangent_force\\":");
            print_scalar_array(left_tangent_force, steps);
            printf(",\\"right_tangent_force\\":");
            print_scalar_array(right_tangent_force, steps);
            printf(",\\"qfrc_constraint_norm\\":");
            print_scalar_array(qfrc_norm, steps);
            printf(",\\"decim\\":%d,\\"dt\\":%.17g,\\"newton\\":%d,\\"ls\\":%d}}\\n", decim, dt, newton, ls);
            return 0;
        }}
        """
    )


def _instrumented_host_source() -> str:
    text = _NANOG1_HOST.read_text()
    text = text.replace(
        "static double qfrc_constraint[HC_NV], qacc_out[HC_NV];",
        "static double qfrc_constraint[HC_NV], qacc_out[HC_NV];\n"
        "static int phx_trace_con_start[HC_NCON_MAX], phx_trace_con_dim[HC_NCON_MAX];\n"
        "static double phx_trace_efc_force[HC_NEFC_MAX];",
        1,
    )
    text = text.replace(
        "int con_start[HC_NCON_MAX], con_dim[HC_NCON_MAX]; double con_fri0[HC_NCON_MAX];",
        "int con_start[HC_NCON_MAX], con_dim[HC_NCON_MAX]; double con_fri0[HC_NCON_MAX];\n"
        "    for (int ci=0; ci<HC_NCON_MAX; ++ci) { con_start[ci]=-1; con_dim[ci]=0; }",
        1,
    )
    text = text.replace(
        "    efc_nefc=n;\n    // impedance -> R,K,B,I ; aref ; pyramidal R adjustment",
        "    efc_nefc=n;\n"
        "    for (int ci=0; ci<HC_NCON_MAX; ++ci) { phx_trace_con_start[ci]=con_start[ci]; phx_trace_con_dim[ci]=con_dim[ci]; }\n"
        "    // impedance -> R,K,B,I ; aref ; pyramidal R adjustment",
        1,
    )
    text = text.replace(
        "if (efc_nefc==0){ memcpy(qacc_out, qacc_smooth, HC_NV*sizeof(double)); memset(qfrc_constraint,0,sizeof qfrc_constraint); return; }",
        "if (efc_nefc==0){ memcpy(qacc_out, qacc_smooth, HC_NV*sizeof(double)); memset(qfrc_constraint,0,sizeof qfrc_constraint); memset(phx_trace_efc_force,0,sizeof phx_trace_efc_force); return; }",
        1,
    )
    text = text.replace(
        "    memcpy(qacc_out, qacc, HC_NV*sizeof(double));\n}",
        "    memset(phx_trace_efc_force,0,sizeof phx_trace_efc_force);\n"
        "    for (int i=0; i<efc_nefc; ++i) phx_trace_efc_force[i]=force[i];\n"
        "    memcpy(qacc_out, qacc, HC_NV*sizeof(double));\n}",
        1,
    )
    return text


def _compile_host_runner(compiler: str, build_dir: Path) -> Path:
    if not _NANOG1_HOST.exists():
        raise FileNotFoundError(f"nanoG1 host stepper not found: {_NANOG1_HOST}")
    host_source = build_dir / "g1_host_instrumented.c"
    host_source.write_text(_instrumented_host_source())
    source = build_dir / "g1_host_runner.c"
    exe = build_dir / "g1_host_runner"
    source.write_text(_host_runner_source(host_source))
    cmd = [compiler, "-O3", "-std=c99", "-I", str(_NANOG1_WEB_DIR), str(source), "-lm", "-o", str(exe)]
    subprocess.run(cmd, check=True, cwd=build_dir, text=True, capture_output=True)
    return exe


def _run_nanog1_host(args: argparse.Namespace, action_pattern: str, action_amplitude: float) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="phoenx_g1_host_") as tmp:
        exe = _compile_host_runner(str(args.c_compiler), Path(tmp))
        cmd = [
            str(exe),
            str(int(args.steps)),
            action_pattern,
            f"{float(action_amplitude):.17g}",
            str(int(args.nanog1_decimation)),
            f"{float(args.nanog1_dt):.17g}",
            str(int(args.nanog1_newton_iterations)),
            str(int(args.nanog1_line_search_iterations)),
            str(args.nanog1_stepper),
            "nan" if args.initial_base_z is None else f"{float(args.initial_base_z):.17g}",
        ]
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)
    return {
        "qpos": np.asarray(payload["qpos"], dtype=np.float64),
        "qvel": np.asarray(payload["qvel"], dtype=np.float64),
        "base_z": np.asarray(payload["base_z"], dtype=np.float64),
        "upright_cos": np.asarray(payload["upright_cos"], dtype=np.float64),
        "foot_contacts": np.stack(
            (
                np.asarray(payload["left_contacts"], dtype=np.float64),
                np.asarray(payload["right_contacts"], dtype=np.float64),
            ),
            axis=1,
        ),
        "foot_normal_force": np.stack(
            (
                np.asarray(payload["left_normal_force"], dtype=np.float64),
                np.asarray(payload["right_normal_force"], dtype=np.float64),
            ),
            axis=1,
        ),
        "foot_tangent_force": np.stack(
            (
                np.asarray(payload["left_tangent_force"], dtype=np.float64),
                np.asarray(payload["right_tangent_force"], dtype=np.float64),
            ),
            axis=1,
        ),
        "qfrc_constraint_norm": np.asarray(payload["qfrc_constraint_norm"], dtype=np.float64),
    }


def _make_phoenx_env(setting: PhoenXSetting, args: argparse.Namespace, device: wp.context.Device) -> rl.EnvG1PhoenX:
    return rl.EnvG1PhoenX(
        rl.ConfigEnvG1PhoenX(
            world_count=1,
            sim_substeps=int(setting.sim_substeps),
            solver_iterations=int(setting.solver_iterations),
            velocity_iterations=int(setting.velocity_iterations),
            joint_friction_model=str(args.joint_friction_model),
            joint_friction_scale=float(args.joint_friction_scale),
            command=(0.0, 0.0, 0.0),
            max_episode_steps=0,
            auto_reset=False,
            parse_meshes=bool(args.parse_meshes),
            contact_geometry=str(args.contact_geometry),
            rigid_contact_max_per_world=int(args.rigid_contact_max_per_world),
            threads_per_world=args.threads_per_world,
            multi_world_scheduler=str(args.multi_world_scheduler),
            prepare_refresh_stride=args.prepare_refresh_stride,
        ),
        device=device,
    )


def _phoenx_qpos_to_nanog1_layout(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[3:7] = (q[6], q[3], q[4], q[5])
    return out


def _run_phoenx(
    setting: PhoenXSetting,
    args: argparse.Namespace,
    action_row: np.ndarray,
    *,
    device: wp.context.Device,
) -> dict[str, np.ndarray]:
    env = _make_phoenx_env(setting, args, device)
    actions = wp.array(action_row.reshape(1, -1), dtype=wp.float32, device=device)
    qpos = np.zeros((int(args.steps) + 1, env.coord_stride), dtype=np.float64)
    qvel = np.zeros((int(args.steps) + 1, env.dof_stride), dtype=np.float64)
    base_z = np.zeros(int(args.steps) + 1, dtype=np.float64)
    upright = np.zeros(int(args.steps) + 1, dtype=np.float64)
    foot_contacts = np.zeros((int(args.steps), 2), dtype=np.float64)
    foot_normal_impulse = np.zeros((int(args.steps), 2), dtype=np.float64)
    foot_tangent_impulse = np.zeros((int(args.steps), 2), dtype=np.float64)
    foot_tangent_normal_ratio = np.zeros((int(args.steps), 2), dtype=np.float64)
    foot_metrics = wp.zeros((env.world_count, 2, G1_FOOT_CONTACT_METRIC_COUNT_TOTAL), dtype=wp.float32, device=device)

    env.reset()
    if args.initial_base_z is not None:
        q_init = env.state_0.joint_q.numpy()
        qd_init = env.state_0.joint_qd.numpy()
        q_init[2] = np.float32(args.initial_base_z)
        qd_init.fill(0.0)
        env.state_0.joint_q.assign(q_init)
        env.state_0.joint_qd.assign(qd_init)
        newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)
        env.observe()
    q0 = env.state_0.joint_q.numpy().reshape(1, env.coord_stride)[0]
    qd0 = env.state_0.joint_qd.numpy().reshape(1, env.dof_stride)[0]
    obs0 = env.obs.numpy()[0]
    qpos[0] = _phoenx_qpos_to_nanog1_layout(q0)
    qvel[0] = qd0
    base_z[0] = q0[2]
    upright[0] = -obs0[5]

    for step in range(int(args.steps)):
        env.step(actions)
        q = env.state_0.joint_q.numpy().reshape(1, env.coord_stride)[0]
        qd = env.state_0.joint_qd.numpy().reshape(1, env.dof_stride)[0]
        obs = env.obs.numpy()[0]
        scan_g1_foot_contact_metrics(env, foot_metrics)
        metrics = foot_metrics.numpy()[0]
        qpos[step + 1] = _phoenx_qpos_to_nanog1_layout(q)
        qvel[step + 1] = qd
        base_z[step + 1] = q[2]
        upright[step + 1] = -obs[5]
        foot_contacts[step] = metrics[:, G1_FOOT_CONTACT_METRIC_COUNT]
        foot_normal_impulse[step] = metrics[:, G1_FOOT_CONTACT_METRIC_NORMAL_IMPULSE]
        foot_tangent_impulse[step] = metrics[:, G1_FOOT_CONTACT_METRIC_TANGENT_IMPULSE]
        counts = np.maximum(metrics[:, G1_FOOT_CONTACT_METRIC_COUNT], 1.0e-12)
        foot_tangent_normal_ratio[step] = metrics[:, G1_FOOT_CONTACT_METRIC_TANGENT_NORMAL_RATIO_SUM] / counts

    return {
        "qpos": qpos,
        "qvel": qvel,
        "base_z": base_z,
        "upright_cos": upright,
        "foot_contacts": foot_contacts,
        "foot_normal_impulse": foot_normal_impulse,
        "foot_tangent_impulse": foot_tangent_impulse,
        "foot_tangent_normal_ratio": foot_tangent_normal_ratio,
        "physics_dt": np.float64(float(env.config.frame_dt) / float(env.config.sim_substeps)),
    }


def _fall_step(base_z: np.ndarray, upright_cos: np.ndarray) -> int | None:
    fallen = (base_z < 0.35) | (upright_cos < 0.6) | ~np.isfinite(base_z) | ~np.isfinite(upright_cos)
    indices = np.nonzero(fallen)[0]
    if indices.size == 0:
        return None
    return int(indices[0])


def _quat_error(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    direct = np.linalg.norm(q - ref, axis=1)
    flipped = np.linalg.norm(q + ref, axis=1)
    return np.minimum(direct, flipped)


def _mean_tracking_ratio_at_step(qpos: np.ndarray, action_row: np.ndarray, step: int) -> float | None:
    controlled_count = int(g1_recipe.CONTROLLED_ACTION_COUNT)
    action_delta = np.float64(g1_recipe.ACTION_SCALE) * action_row[:controlled_count].astype(np.float64)
    active = np.abs(action_delta) > 1.0e-7
    if not np.any(active):
        return None
    ratio = (qpos[step, 7 : 7 + controlled_count][active] - qpos[0, 7 : 7 + controlled_count][active]) / action_delta[
        active
    ]
    return float(np.mean(ratio))


def _trace_grounded_divergence(
    phoenx: dict[str, np.ndarray],
    host: dict[str, np.ndarray],
    action_row: np.ndarray,
    trace_steps: int,
) -> list[dict[str, float | int | None]]:
    count = min(int(trace_steps), int(phoenx["foot_contacts"].shape[0]), int(host["foot_contacts"].shape[0]))
    if count <= 0:
        return []

    q = phoenx["qpos"]
    qd = phoenx["qvel"]
    hq = host["qpos"]
    hqd = host["qvel"]
    trace: list[dict[str, float | int | None]] = []
    for row in range(count):
        step = row + 1
        joint_err = q[step, 7 : 7 + rl.ACTION_DIM_G1] - hq[step, 7 : 7 + rl.ACTION_DIM_G1]
        joint_qd_err = qd[step, 6 : 6 + rl.ACTION_DIM_G1] - hqd[step, 6 : 6 + rl.ACTION_DIM_G1]
        base_xy_err = q[step, 0:2] - hq[step, 0:2]
        phoenx_tracking = _mean_tracking_ratio_at_step(q, action_row, step)
        host_tracking = _mean_tracking_ratio_at_step(hq, action_row, step)
        tracking_delta = None
        if phoenx_tracking is not None and host_tracking is not None:
            tracking_delta = float(phoenx_tracking - host_tracking)
        phoenx_contacts = phoenx["foot_contacts"][row]
        host_contacts = host["foot_contacts"][row]
        support_normal = float(np.sum(phoenx["foot_normal_impulse"][row]))
        support_tangent = float(np.sum(phoenx["foot_tangent_impulse"][row]))
        physics_dt = float(phoenx["physics_dt"])
        nanog1_support_normal = float(np.sum(host["foot_normal_force"][row]))
        nanog1_support_tangent = float(np.sum(host["foot_tangent_force"][row]))
        trace.append(
            {
                "step": step,
                "phoenx_base_z_m": float(phoenx["base_z"][step]),
                "nanog1_base_z_m": float(host["base_z"][step]),
                "base_z_delta_m": float(phoenx["base_z"][step] - host["base_z"][step]),
                "base_xy_delta_m": float(np.linalg.norm(base_xy_err)),
                "phoenx_upright_cos": float(phoenx["upright_cos"][step]),
                "nanog1_upright_cos": float(host["upright_cos"][step]),
                "upright_cos_delta": float(phoenx["upright_cos"][step] - host["upright_cos"][step]),
                "joint_q_rmse_rad": float(np.sqrt(np.mean(joint_err * joint_err))),
                "joint_qd_rmse_rad_s": float(np.sqrt(np.mean(joint_qd_err * joint_qd_err))),
                "phoenx_tracking_ratio_mean": phoenx_tracking,
                "nanog1_tracking_ratio_mean": host_tracking,
                "tracking_ratio_delta_mean": tracking_delta,
                "phoenx_left_contact_count": float(phoenx_contacts[0]),
                "phoenx_right_contact_count": float(phoenx_contacts[1]),
                "nanog1_left_contact_count": float(host_contacts[0]),
                "nanog1_right_contact_count": float(host_contacts[1]),
                "left_contact_count_delta": float(phoenx_contacts[0] - host_contacts[0]),
                "right_contact_count_delta": float(phoenx_contacts[1] - host_contacts[1]),
                "phoenx_support_normal_impulse": support_normal,
                "phoenx_support_tangent_impulse": support_tangent,
                "phoenx_support_normal_force_est": float(support_normal / physics_dt),
                "phoenx_support_tangent_force_est": float(support_tangent / physics_dt),
                "phoenx_support_tangent_normal_ratio": float(support_tangent / max(support_normal, 1.0e-12)),
                "phoenx_foot_tangent_normal_ratio_mean": float(np.mean(phoenx["foot_tangent_normal_ratio"][row])),
                "nanog1_support_normal_force": nanog1_support_normal,
                "nanog1_support_tangent_force": nanog1_support_tangent,
                "nanog1_support_tangent_normal_ratio": float(
                    nanog1_support_tangent / max(nanog1_support_normal, 1.0e-12)
                ),
                "nanog1_qfrc_constraint_norm": float(host["qfrc_constraint_norm"][row]),
            }
        )
    return trace


def _compare_trajectory(
    setting: PhoenXSetting,
    phoenx: dict[str, np.ndarray],
    host: dict[str, np.ndarray],
    action_row: np.ndarray,
    trace_steps: int = 0,
) -> dict[str, Any]:
    q = phoenx["qpos"]
    qd = phoenx["qvel"]
    hq = host["qpos"]
    hqd = host["qvel"]
    joint_err = q[:, 7 : 7 + rl.ACTION_DIM_G1] - hq[:, 7 : 7 + rl.ACTION_DIM_G1]
    joint_qd_err = qd[:, 6 : 6 + rl.ACTION_DIM_G1] - hqd[:, 6 : 6 + rl.ACTION_DIM_G1]
    base_pos_err = q[:, 0:3] - hq[:, 0:3]
    base_quat_err = _quat_error(q[:, 3:7], hq[:, 3:7])
    base_z_err = phoenx["base_z"] - host["base_z"]
    upright_err = phoenx["upright_cos"] - host["upright_cos"]
    foot_contact_err = phoenx["foot_contacts"] - host["foot_contacts"]
    physics_dt = float(g1_recipe.FRAME_DT) / float(setting.sim_substeps)
    phoenx_support_normal_impulse = np.sum(phoenx["foot_normal_impulse"], axis=1)
    phoenx_support_tangent_impulse = np.sum(phoenx["foot_tangent_impulse"], axis=1)
    nanog1_support_normal_force = np.sum(host["foot_normal_force"], axis=1)
    nanog1_support_tangent_force = np.sum(host["foot_tangent_force"], axis=1)
    controlled_count = int(g1_recipe.CONTROLLED_ACTION_COUNT)
    action_delta = np.float64(g1_recipe.ACTION_SCALE) * action_row[:controlled_count].astype(np.float64)
    target = hq[0, 7 : 7 + controlled_count] + action_delta
    host_final_error = hq[-1, 7 : 7 + controlled_count] - target
    phoenx_final_error = q[-1, 7 : 7 + controlled_count] - target
    target_error_delta = phoenx_final_error - host_final_error
    active = np.abs(action_delta) > 1.0e-7
    host_tracking_ratio = None
    phoenx_tracking_ratio = None
    tracking_ratio_delta = None
    if np.any(active):
        host_ratio = (
            hq[-1, 7 : 7 + controlled_count][active] - hq[0, 7 : 7 + controlled_count][active]
        ) / action_delta[active]
        phoenx_ratio = (
            q[-1, 7 : 7 + controlled_count][active] - q[0, 7 : 7 + controlled_count][active]
        ) / action_delta[active]
        host_tracking_ratio = float(np.mean(host_ratio))
        phoenx_tracking_ratio = float(np.mean(phoenx_ratio))
        tracking_ratio_delta = float(phoenx_tracking_ratio - host_tracking_ratio)
    result: dict[str, Any] = {
        "setting": setting.name,
        "sim_substeps": int(setting.sim_substeps),
        "solver_iterations": int(setting.solver_iterations),
        "velocity_iterations": int(setting.velocity_iterations),
        "physics_dt": physics_dt,
        "phoenx_fall_step": _fall_step(phoenx["base_z"], phoenx["upright_cos"]),
        "nanog1_fall_step": _fall_step(host["base_z"], host["upright_cos"]),
        "phoenx_final_base_z_m": float(phoenx["base_z"][-1]),
        "nanog1_final_base_z_m": float(host["base_z"][-1]),
        "phoenx_final_upright_cos": float(phoenx["upright_cos"][-1]),
        "nanog1_final_upright_cos": float(host["upright_cos"][-1]),
        "base_z_rmse_m": float(np.sqrt(np.mean(base_z_err * base_z_err))),
        "base_z_max_abs_m": float(np.max(np.abs(base_z_err))),
        "upright_cos_rmse": float(np.sqrt(np.mean(upright_err * upright_err))),
        "base_pos_final_error_m": float(np.linalg.norm(base_pos_err[-1])),
        "base_pos_rmse_m": float(np.sqrt(np.mean(np.sum(base_pos_err * base_pos_err, axis=1)))),
        "base_quat_final_error": float(base_quat_err[-1]),
        "base_quat_rmse": float(np.sqrt(np.mean(base_quat_err * base_quat_err))),
        "joint_q_final_rmse_rad": float(np.sqrt(np.mean(joint_err[-1] * joint_err[-1]))),
        "joint_q_traj_rmse_rad": float(np.sqrt(np.mean(joint_err * joint_err))),
        "joint_q_final_max_abs_rad": float(np.max(np.abs(joint_err[-1]))),
        "joint_qd_final_rmse_rad_s": float(np.sqrt(np.mean(joint_qd_err[-1] * joint_qd_err[-1]))),
        "joint_qd_traj_rmse_rad_s": float(np.sqrt(np.mean(joint_qd_err * joint_qd_err))),
        "phoenx_final_target_error_rmse_rad": float(np.sqrt(np.mean(phoenx_final_error * phoenx_final_error))),
        "nanog1_final_target_error_rmse_rad": float(np.sqrt(np.mean(host_final_error * host_final_error))),
        "target_error_final_delta_rmse_rad": float(np.sqrt(np.mean(target_error_delta * target_error_delta))),
        "phoenx_tracking_ratio_mean": phoenx_tracking_ratio,
        "nanog1_tracking_ratio_mean": host_tracking_ratio,
        "tracking_ratio_delta_mean": tracking_ratio_delta,
        "left_contact_count_final_error": float(foot_contact_err[-1, 0]),
        "right_contact_count_final_error": float(foot_contact_err[-1, 1]),
        "contact_count_traj_rmse": float(np.sqrt(np.mean(foot_contact_err * foot_contact_err))),
        "phoenx_left_contact_count_mean": float(np.mean(phoenx["foot_contacts"][:, 0])),
        "phoenx_right_contact_count_mean": float(np.mean(phoenx["foot_contacts"][:, 1])),
        "nanog1_left_contact_count_mean": float(np.mean(host["foot_contacts"][:, 0])),
        "nanog1_right_contact_count_mean": float(np.mean(host["foot_contacts"][:, 1])),
        "phoenx_foot_normal_impulse_mean": float(np.mean(phoenx["foot_normal_impulse"])),
        "phoenx_foot_tangent_impulse_mean": float(np.mean(phoenx["foot_tangent_impulse"])),
        "phoenx_support_normal_force_est_mean": float(np.mean(phoenx_support_normal_impulse / physics_dt)),
        "phoenx_support_tangent_force_est_mean": float(np.mean(phoenx_support_tangent_impulse / physics_dt)),
        "phoenx_foot_tangent_normal_ratio_mean": float(np.mean(phoenx["foot_tangent_normal_ratio"])),
        "nanog1_support_normal_force_mean": float(np.mean(nanog1_support_normal_force)),
        "nanog1_support_tangent_force_mean": float(np.mean(nanog1_support_tangent_force)),
        "nanog1_support_tangent_normal_ratio_mean": float(
            np.mean(nanog1_support_tangent_force / np.maximum(nanog1_support_normal_force, 1.0e-12))
        ),
        "nanog1_qfrc_constraint_norm_mean": float(np.mean(host["qfrc_constraint_norm"])),
    }
    if int(trace_steps) > 0:
        result["trace"] = _trace_grounded_divergence(phoenx, host, action_row, int(trace_steps))
    return result


def benchmark_open_loop_parity(args: argparse.Namespace) -> dict[str, Any]:
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("G1 open-loop parity benchmark requires CUDA for PhoenX")
    if int(args.steps) <= 0:
        raise ValueError("steps must be positive")
    action_row = _leg_action_pattern(str(args.action_pattern), float(args.action_amplitude))
    host = _run_nanog1_host(args, str(args.action_pattern), float(args.action_amplitude))
    results = []
    for name in tuple(args.settings):
        setting = _SETTINGS[name]
        phoenx = _run_phoenx(setting, args, action_row, device=device)
        results.append(_compare_trajectory(setting, phoenx, host, action_row, int(args.trace_steps)))
    return {
        "engine": "phoenx_vs_nanog1_host_open_loop",
        "metric": "same reset and open-loop action targets, state error, contact support, and drive tracking response",
        "device": device.name,
        "steps": int(args.steps),
        "action_pattern": str(args.action_pattern),
        "action_amplitude": float(args.action_amplitude),
        "contact_geometry": str(args.contact_geometry),
        "joint_friction_model": str(args.joint_friction_model),
        "joint_friction_scale": float(args.joint_friction_scale),
        "initial_base_z": None if args.initial_base_z is None else float(args.initial_base_z),
        "nanog1_stepper": str(args.nanog1_stepper),
        "trace_steps": int(args.trace_steps),
        "nanog1_reference": {
            "host_stepper": str(_NANOG1_HOST),
            "dt": float(args.nanog1_dt),
            "decimation": int(args.nanog1_decimation),
            "newton_iterations": int(args.nanog1_newton_iterations),
            "line_search_iterations": int(args.nanog1_line_search_iterations),
            "stepper": str(args.nanog1_stepper),
            "action_scale": 0.25,
            "unitree_pd": True,
        },
        "results": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--action-pattern", choices=("zero", "leg_step", "leg_symmetric"), default="zero")
    parser.add_argument("--action-amplitude", type=float, default=0.2)
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=0,
        help="Include detailed per-control-step trace records for the first N steps.",
    )
    parser.add_argument("--settings", type=_parse_csv, default=("fast_5x2", "recipe_default", "phoenx_10x8"))
    parser.add_argument("--nanog1-dt", type=float, default=0.004)
    parser.add_argument("--nanog1-decimation", type=int, default=5)
    parser.add_argument("--nanog1-newton-iterations", type=int, default=2)
    parser.add_argument("--nanog1-line-search-iterations", type=int, default=3)
    parser.add_argument(
        "--nanog1-stepper",
        choices=("full", "smooth"),
        default="full",
        help="nanoG1 host stepper: full constraints/friction or smooth dynamics only.",
    )
    parser.add_argument(
        "--initial-base-z",
        type=float,
        default=None,
        help="Override reset base height [m], useful for contact-free drive response sweeps.",
    )
    parser.add_argument("--joint-friction-model", choices=("hard", "mujoco"), default=g1_recipe.JOINT_FRICTION_MODEL)
    parser.add_argument("--joint-friction-scale", type=float, default=g1_recipe.JOINT_FRICTION_SCALE)
    parser.add_argument("--parse-meshes", action="store_true")
    parser.add_argument("--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=g1_recipe.CONTACT_GEOMETRY)
    parser.add_argument("--rigid-contact-max-per-world", type=int, default=g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
    parser.add_argument("--threads-per-world", default=g1_recipe.THREADS_PER_WORLD)
    parser.add_argument("--multi-world-scheduler", default=g1_recipe.MULTI_WORLD_SCHEDULER)
    parser.add_argument("--prepare-refresh-stride", default=g1_recipe.PREPARE_REFRESH_STRIDE)
    parser.add_argument("--c-compiler", default=os.environ.get("CC", "cc"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = benchmark_open_loop_parity(args)
    print(json.dumps(result, indent=args.json_indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
