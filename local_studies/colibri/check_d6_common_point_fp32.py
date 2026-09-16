# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Offline FP32 six-row wrench/scatter audit; not an actual kernel dispatch test."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation  # noqa: TID253


def main():
    rng = np.random.default_rng(8125)
    maxima = {"unit_linear": 0.0, "unit_angular": 0.0, "scatter_linear": 0.0, "scatter_angular": 0.0, "midpoint_jacobian": 0.0}
    negative = []
    for _ in range(200):
        p = rng.uniform(-0.2, 0.2, (2, 3)).astype(np.float32)
        common = rng.uniform(-0.1, 0.1, 3).astype(np.float32)
        separation = rng.normal(size=3)
        separation = (separation / np.linalg.norm(separation) * 58e-6).astype(np.float32)
        lever0 = (common - p[0] - np.float32(0.5) * separation).astype(np.float32)
        lever1 = (common - p[1] + np.float32(0.5) * separation).astype(np.float32)
        error = (p[1] - p[0] + lever1 - lever0).astype(np.float32)
        corrected0 = lever0 + np.float32(0.5) * error
        corrected1 = lever1 - np.float32(0.5) * error
        axes = Rotation.random(random_state=rng).as_matrix().astype(np.float32)
        j = np.zeros((6, 2, 6), np.float32)
        old = np.zeros_like(j)
        for row in range(3):
            axis = axes[:, row]
            j[row, 0, :3], j[row, 1, :3] = -axis, axis
            j[row, 0, 3:] = np.cross(corrected0, -axis)
            j[row, 1, 3:] = np.cross(corrected1, axis)
            old[row] = j[row]
            old[row, 0, 3:] = np.cross(lever0, -axis)
            old[row, 1, 3:] = np.cross(lever1, axis)
            j[row + 3, 0, 3:], j[row + 3, 1, 3:] = -axis, axis
        physical_point = 0.5 * (p[0].astype(float) + lever0 + p[1].astype(float) + lever1)
        exact = j.astype(float)
        for row in range(3):
            for body in range(2):
                exact[row, body, 3:] = np.cross(physical_point - p[body], exact[row, body, :3])
        maxima["midpoint_jacobian"] = max(maxima["midpoint_jacobian"], float(np.max(abs(exact - j))))
        masses = np.array([0.31, 0.79], np.float32)
        inertia = []
        for _body in range(2):
            r = Rotation.random(random_state=rng).as_matrix()
            inertia.append((r @ np.diag([0.003, 0.005, 0.008]) @ r.T).astype(np.float32))
        for row in range(6):
            force = j[row, :, :3].astype(float)
            torque = j[row, :, 3:].astype(float)
            maxima["unit_linear"] = max(maxima["unit_linear"], float(np.max(abs(force.sum(axis=0)))))
            maxima["unit_angular"] = max(
                maxima["unit_angular"], float(np.max(abs((torque + np.cross(p, force)).sum(axis=0))))
            )
            impulse = np.float32(rng.uniform(-0.003, 0.003))
            dv = (j[row, :, :3] * impulse / masses[:, None]).astype(np.float32)
            dw = np.stack([np.linalg.solve(inertia[b], j[row, b, 3:] * impulse) for b in range(2)])
            linear = masses[:, None].astype(float) * dv
            angular = np.stack([inertia[b].astype(float) @ dw[b] for b in range(2)]) + np.cross(p, linear)
            maxima["scatter_linear"] = max(maxima["scatter_linear"], float(np.max(abs(linear.sum(axis=0)))))
            maxima["scatter_angular"] = max(maxima["scatter_angular"], float(np.max(abs(angular.sum(axis=0)))))
        old_torque = old[:3, :, 3:].astype(float) + np.cross(p, old[:3, :, :3].astype(float))
        negative.append(float(np.max(abs(old_torque.sum(axis=1)))))
    assert maxima["unit_linear"] == 0
    assert maxima["unit_angular"] < 2e-7
    assert maxima["scatter_linear"] < 2e-9
    assert maxima["scatter_angular"] < 2e-9
    assert maxima["midpoint_jacobian"] < 1e-7
    assert min(negative) > 1e-5
    result = {
        "cases": 200,
        "rows_per_case": 6,
        "anchor_separation_m": 58e-6,
        "candidate_dtype": "float32",
        "scope": "Independent source-equation/scatter reference, not actual kernel dispatch",
        "maxima": maxima,
        "separated_anchor_negative_min_unit_torque_m": min(negative),
    }
    Path("/tmp/colibri_d6_common_point_fp32.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
