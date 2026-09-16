# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent quaternion-formula audit of the pose-only closure candidate."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np


def rotate(quaternion, vector):
    """Apply a normalized quaternion without the optimizer's rotation library."""
    q = np.asarray(quaternion, dtype=float)
    q = q / np.linalg.norm(q)
    v = np.asarray(vector, dtype=float)
    t = 2 * np.cross(q[:3], v)
    return v + q[3] * t + np.cross(q[:3], t)


def main():
    """Check every original joint frame and the fixed body poses."""
    path = Path("newton/examples/kamino/example_kamino_colibri.py")
    source = path.read_bytes()
    constants = {}
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("BODY_POSES", "JOINTS")
        ):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    data = np.load("/tmp/kamino_initial_joint_closure.npz")
    labels = data["body_labels"].tolist()
    pose, original = data["pose"], data["initial_pose"]
    checks = []
    for joint in constants["JOINTS"]:
        a, b, kind, axis, fa, fb = joint[:6]
        ia, ib = labels.index(a), labels.index(b)
        pa = pose[ia, :3] + rotate(pose[ia, 3:], fa[:3])
        pb = pose[ib, :3] + rotate(pose[ib, 3:], fb[:3])
        direction = np.eye(3)["XYZ".index(axis)]
        wa = rotate(pose[ia, 3:], rotate(fa[3:], direction))
        wb = rotate(pose[ib, 3:], rotate(fb[3:], direction))
        angle = float(np.arctan2(np.linalg.norm(np.cross(wa, wb)), wa @ wb)) if kind == "revolute" else 0
        checks.append({"pair": [a, b], "anchor_m": float(np.linalg.norm(pa - pb)), "axis_rad": angle})
    for i, label in enumerate(labels):
        if label in constants["BODY_POSES"]:
            assert np.array_equal(original[i], np.asarray(constants["BODY_POSES"][label]))
    fixed = [labels.index(label) for label in ("FrameGround", "Flower", "TailRack")]
    assert np.array_equal(pose[fixed], original[fixed])
    summary = {
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "joint_count": len(checks),
        "max_anchor_m": max(value["anchor_m"] for value in checks),
        "max_axis_rad": max(value["axis_rad"] for value in checks),
        "fixed_bodies_unchanged": True,
        "source_initial_poses_exact": True,
        "scope": "Independent quaternion formula and all authored joint frames; no contact or live acceptance",
    }
    assert summary["max_anchor_m"] < 1e-10 and summary["max_axis_rad"] < 1e-10
    Path("/tmp/colibri_closure_independent_audit.json").write_text(
        json.dumps({"summary": summary, "joints": checks}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
