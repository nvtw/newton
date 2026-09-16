# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Describe long-run phase coherence without assuming authored gear ratios."""

import argparse
import json
from pathlib import Path

import numpy as np

from local_studies.colibri.drive_motion import multiply
from newton.examples.kamino.example_kamino_colibri import JOINTS


def analyze(snapshot):
    """Fit observed rotation ratios; do not interpret them as design validation."""
    with np.load(snapshot) as data:
        history = data["q_history"].astype(np.float64)
        times = data["history_times"].astype(np.float64)
        labels = data["labels"].astype(str).tolist()
    indices = {label: index for index, label in enumerate(labels)}
    angles = {}
    for parent, child, kind, axis, fa, fb, _ in JOINTS:
        if kind != "revolute" or parent != "Frame":
            continue
        qa = multiply(history[:, indices[parent], 3:], fa[3:])
        qb = multiply(history[:, indices[child], 3:], fb[3:])
        qa[:, :3] *= -1.0
        relative = multiply(qa, qb)
        angles[child] = np.unwrap(2 * np.arctan2(relative[:, "XYZ".index(axis)], relative[:, 3]))
    crank = angles["Crank"]
    mask = times >= times[0] + 1.0
    x = crank[mask]
    design = np.column_stack((x, np.ones_like(x)))
    rows = []
    for name, angle in angles.items():
        if not (name.startswith("Gear_Large") or name.startswith("CamWheel") or name == "GearedSpinner"):
            continue
        ratio, intercept = np.linalg.lstsq(design, angle[mask], rcond=None)[0]
        residual = angle[mask] - ratio * x - intercept
        half = len(residual) // 2
        interval_ratios = []
        for segment in np.array_split(np.arange(len(x)), 10):
            interval_ratios.append(
                float((angle[mask][segment[-1]] - angle[mask][segment[0]]) / (x[segment[-1]] - x[segment[0]]))
            )
        rows.append(
            {
                "joint": f"Frame/{name}",
                "observed_fitted_ratio_to_crank": float(ratio),
                "phase_residual_peak_to_peak_rad": float(np.ptp(residual)),
                "phase_residual_rms_rad": float(np.sqrt(np.mean(residual**2))),
                "phase_residual_half_mean_shift_rad": float(np.mean(residual[half:]) - np.mean(residual[:half])),
                "ten_interval_net_rotation_ratios": interval_ratios,
            }
        )
    speed = np.diff(crank) / np.diff(times)
    return {
        "snapshot": str(snapshot),
        "samples": len(times),
        "duration_s": float(times[-1] - times[0]),
        "fit_excludes_startup_s": 1.0,
        "scope": "Observed phase coherence only. Fitted ratios are not independently verified design ratios; linear slip can be absorbed by the fit.",
        "crank_mean_speed_rad_s": float((crank[-1] - crank[0]) / (times[-1] - times[0])),
        "crank_max_speed_rad_s": float(np.max(speed)),
        "crank_min_speed_rad_s": float(np.min(speed)),
        "joints": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.snapshot)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
