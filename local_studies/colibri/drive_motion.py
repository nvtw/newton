# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Audit revolute motion from saved body orientations and authored joint frames."""

import argparse
import json
from pathlib import Path

import numpy as np

from newton.examples.kamino.example_kamino_colibri import BODY_ORDER, JOINTS


def multiply(a, b):
    """Multiply broadcastable XYZW quaternions in FP64."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    xyz = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + np.cross(a[..., :3], b[..., :3])
    w = a[..., 3:] * b[..., 3:] - np.sum(a[..., :3] * b[..., :3], axis=-1, keepdims=True)
    return np.concatenate((xyz, w), axis=-1)


def measure(history, labels, *, fps):
    """Measure unwrapped joint twist; report motion without assuming gear ratios."""
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 3 or history.shape[-1] != 7 or len(history) < 2:
        raise ValueError("Expected at least two frames of body poses")
    indices = {label: index for index, label in enumerate(labels)}
    rows = []
    for parent, child, kind, axis, fa, fb, drive in JOINTS:
        if kind != "revolute":
            continue
        qa = multiply(history[:, indices[parent], 3:], fa[3:])
        qb = multiply(history[:, indices[child], 3:], fb[3:])
        qa[:, :3] *= -1.0
        relative = multiply(qa, qb)
        angle = np.unwrap(2.0 * np.arctan2(relative[:, "XYZ".index(axis)], relative[:, 3]))
        speed = np.diff(angle) * fps
        row = {
            "joint": f"{parent}/{child}",
            "net_turns": float((angle[-1] - angle[0]) / (2.0 * np.pi)),
            "angle_span_rad": float(np.ptp(angle)),
            "mean_speed_rad_s": float(np.mean(speed)),
            "speed_percentiles_rad_s": np.percentile(speed, [0, 5, 50, 95, 100]).tolist(),
            "fraction_below_0_01_rad_s": float(np.mean(np.abs(speed) < 0.01)),
        }
        if "target_vel" in drive:
            row["target_speed_rad_s"] = drive["target_vel"]
        rows.append(row)
    return {"samples": len(history), "duration_s": (len(history) - 1) / fps, "joints": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.snapshot)
    history = data["q_history"] if "q_history" in data else data["q"]
    labels = data["labels"].astype(str).tolist() if "labels" in data else [*BODY_ORDER, "Flower"]
    if args.seconds is not None:
        history = history[: round(args.seconds * args.fps) + 1]
    result = {"snapshot": str(args.snapshot), **measure(history, labels, fps=args.fps)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "joints": [r for r in result["joints"] if "target_speed_rad_s" in r]}, indent=2))


if __name__ == "__main__":
    main()
