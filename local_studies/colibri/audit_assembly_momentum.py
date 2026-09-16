# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure aggregate Colibri motion from saved SI poses and COM velocities."""

import argparse
import json
from pathlib import Path

import numpy as np


def rotate(q, v):
    """Rotate broadcastable vectors by normalized XYZW quaternions."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    t = 2.0 * np.cross(q[..., :3], v)
    return v + q[..., 3:] * t + np.cross(q[..., :3], t)


def main():
    """Report aggregate momentum and energy without assuming conservation under external loads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = np.load(args.snapshot)
    labels = data["labels"].astype(str)
    dynamic = labels != "Flower"
    q = data["q_history"][:, dynamic].astype(np.float64)
    qd = data["qd_history"][:, dynamic].astype(np.float64)
    times = data["history_times"]
    mass = data["body_mass"][dynamic].astype(np.float64)
    inertia = data["body_inertia"][dynamic].astype(np.float64)
    com_local = data["body_com"][dynamic].astype(np.float64)
    com = q[..., :3] + rotate(q[..., 3:], com_local)
    inverse = q[..., 3:].copy()
    inverse[..., :3] *= -1.0
    angular_local = rotate(inverse, qd[..., 3:])
    spin_local = np.einsum("bij,tbj->tbi", inertia, angular_local)
    linear_body = mass[None, :, None] * qd[..., :3]
    linear = linear_body.sum(axis=1)
    angular = (rotate(q[..., 3:], spin_local) + np.cross(com, linear_body)).sum(axis=1)
    kinetic = 0.5 * (
        mass[None, :] * np.sum(qd[..., :3] ** 2, axis=-1) + np.sum(angular_local * spin_local, axis=-1)
    ).sum(axis=1)
    potential = (mass[None, :] * com[..., 2]).sum(axis=1)  # Authored gravity: 1 m/s².
    center = (mass[None, :, None] * com).sum(axis=1) / mass.sum()
    rows = []
    for start in range(0, int(times[-1]) + 1, 20):
        selected = (times >= start) & (times < start + 20)
        if not np.any(selected):
            continue
        rows.append(
            {
                "start_s": start,
                "end_s": float(times[selected][-1]),
                "mean_linear_momentum_kg_m_s": linear[selected].mean(axis=0).tolist(),
                "mean_angular_momentum_kg_m2_s": angular[selected].mean(axis=0).tolist(),
                "kinetic_energy_J_min_mean_max": [
                    float(kinetic[selected].min()),
                    float(kinetic[selected].mean()),
                    float(kinetic[selected].max()),
                ],
                "mean_potential_energy_J": float(potential[selected].mean()),
            }
        )
    report = {
        "snapshot": str(args.snapshot),
        "dynamic_mass_kg": float(mass.sum()),
        "excluded_kinematic_bodies": labels[~dynamic].tolist(),
        "com_displacement_m": (center[-1] - center[0]).tolist(),
        "scope": "Trajectory diagnostic, not a conservation test: ground/Flower impulses and drive work are not recorded.",
        "windows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
