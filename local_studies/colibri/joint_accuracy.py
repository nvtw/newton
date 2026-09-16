# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure source joint anchors and axes on saved SI body poses."""

import argparse
import json
from pathlib import Path

import numpy as np

from newton.examples.kamino.example_kamino_colibri import BODY_ORDER, JOINTS


def rotate(quaternion, vector):
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    return v + 2.0 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)


def measure(poses, labels):
    """Return independent geometric residuals for the authored enabled joints."""
    body = dict(zip(labels, np.asarray(poses, dtype=np.float64), strict=True))
    rows = []
    for parent, child, kind, axis, frame_parent, frame_child, _drive in JOINTS:
        if parent not in body or child not in body:
            continue
        qa, qb = body[parent], body[child]
        anchor_a = qa[:3] + rotate(qa[3:], frame_parent[:3])
        anchor_b = qb[:3] + rotate(qb[3:], frame_child[:3])
        row = {"joint": f"{parent}/{child}", "anchor_error_m": float(np.linalg.norm(anchor_a - anchor_b))}
        if kind == "revolute":
            basis = np.eye(3)["XYZ".index(axis)]
            a = rotate(qa[3:], rotate(frame_parent[3:], basis))
            b = rotate(qb[3:], rotate(frame_child[3:], basis))
            row["axis_error_rad"] = float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))
        rows.append(row)
    return {
        "joint_count": len(rows),
        "max_anchor_error_m": max((r["anchor_error_m"] for r in rows), default=0.0),
        "max_axis_error_rad": max((r.get("axis_error_rad", 0.0) for r in rows), default=0.0),
        "joints": sorted(rows, key=lambda r: r["anchor_error_m"], reverse=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--physx-centimeters", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.snapshot)
    poses = data["q"].astype(np.float64)
    if args.physx_centimeters:
        poses[:, :3] *= 0.01
        labels = [str(p).rsplit("/", 1)[-1] for p in data["paths"]]
    elif "labels" in data:
        labels = [str(p) for p in data["labels"]]
    else:
        if len(poses) != len(BODY_ORDER) + 1:
            raise ValueError("Unlabelled Newton snapshots must contain the full mechanism plus Flower")
        labels = [*BODY_ORDER, "Flower"]
    result = {"snapshot": str(args.snapshot), **measure(poses, labels)}
    if result["joint_count"] != len(JOINTS):
        raise ValueError("Snapshot does not cover every source joint")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "joints": result["joints"][:5]}, indent=2))


if __name__ == "__main__":
    main()
