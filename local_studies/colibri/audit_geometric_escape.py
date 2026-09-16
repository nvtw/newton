# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU audit of the saved 203-second absolute-travel assertion failure."""

import json
from pathlib import Path

import numpy as np


def main():
    """Separate assembly-relative motion from the base's global motion."""
    from scipy.spatial.transform import Rotation

    data = np.load("/tmp/colibri_public_geometric18000.npz")
    q = np.concatenate((data["q_history"], data["q"][None])).astype(np.float64)
    labels = data["labels"].tolist()
    base = labels.index("FrameGround")
    rotation = Rotation.from_quat(q[:, base, 3:]).as_matrix()
    relative = np.einsum("tji,tbj->tbi", rotation, q[:, :, :3] - q[:, base : base + 1, :3])
    relative_displacement = np.linalg.norm(relative - relative[0], axis=2)
    displacement = np.linalg.norm(q[:, :, :3] - q[0, :, :3], axis=2)
    active = np.array([name != "Flower" for name in labels])
    # Rotate the original base's world-up vector: yaw leaves this unchanged.
    relative_rotation = rotation @ rotation[0].T
    tilt = np.arccos(np.clip(relative_rotation[:, 2, 2], -1.0, 1.0))
    yaw = np.unwrap(np.arctan2(relative_rotation[:, 1, 0], relative_rotation[:, 0, 0]))
    report = {
        "reference": "First saved post-step pose; authored initial pose absent from this artifact",
        "classification": "Absolute-travel bound crossed while assembly remains attached; support drift accuracy unresolved",
        "base_final_translation_m": (q[-1, base, :3] - q[0, base, :3]).tolist(),
        "base_height_range_m": [float(q[:, base, 2].min()), float(q[:, base, 2].max())],
        "base_peak_tilt_from_first_rad": float(tilt.max()),
        "base_final_yaw_from_first_rad": float(yaw[-1]),
        "peak_base_relative_displacement_m": float(relative_displacement[:, active].max()),
        "first_20s_peak_base_relative_displacement_m": float(relative_displacement[:1200, active].max()),
        "max_tail_support_separation_m": float(
            np.linalg.norm(q[:, labels.index("TailRack"), :3] - q[:, labels.index("TailMount"), :3], axis=1).max()
        ),
        "final_max_linear_speed_m_s": float(np.linalg.norm(data["qd"][:, :3], axis=1).max()),
        "final_max_angular_speed_rad_s": float(np.linalg.norm(data["qd"][:, 3:], axis=1).max()),
        "bodies": [
            {
                "name": name,
                "final_absolute_displacement_m": float(displacement[-1, i]),
                "final_relative_displacement_m": float(relative_displacement[-1, i]),
                "peak_relative_displacement_m": float(relative_displacement[:, i].max()),
            }
            for i, name in enumerate(labels)
        ],
        "limitation": "Base and ground friction are both 0.5. Smooth drift may be physical walking or numerical creep; joint attachment alone does not resolve this.",
    }
    original = json.loads(Path("/tmp/colibri_public_geometric18000.json").read_text())
    report["original_assertion_passed"] = False
    report["original_failure"] = original["failure"]
    report["peak_joint_anchor_m"] = original["peak_anchor_m"]
    report["peak_joint_axis_rad"] = original["peak_axis_rad"]
    Path("/tmp/colibri_geometric_escape_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "bodies"}, indent=2))


if __name__ == "__main__":
    main()
