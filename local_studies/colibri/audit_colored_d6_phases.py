# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Evaluate original phase equations using captured physical COM velocities."""

import argparse
import json
from pathlib import Path

import numpy as np

from .check_colored_d6_fp32 import original_residuals
from .coupled_support_online import assemble_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    z = np.load(args.snapshot)
    result = {"scope": "Original captured phase equations; physical bodies, no copy averaging surrogate", "phases": {}}
    for label in (
        "biased_before",
        "biased_after",
        "integrate_before",
        "integrate_after",
        "relax_before",
        "relax_after",
    ):
        d = {key.split(".", 1)[1]: z[key] for key in z.files if key.startswith(label + ".")}
        if not np.any(d["joint_row_count"]):
            result["phases"][label] = {"rejected": "Unwritten phase buffer"}
            continue
        # With splitting disabled, these are the actual solved velocities.
        d["copy_section_end"] = np.arange(len(d["velocity"]), dtype=np.int32)
        d["copy_velocity"] = d["velocity"][1:]
        d["copy_angular_velocity"] = d["angular_velocity"][1:]
        phase = "relax" if label.startswith("relax") else "biased"
        a = assemble_snapshot(d, phase, float(z["dt"][0]), int(z["num_joints"][0]))
        bias = a["rhs"] - a["C"] @ a["vbar"]
        residual = original_residuals(a, a["velocity"], a["old_joint"], a["old"], bias)
        triples = a["old"].reshape(-1, 3)
        jv = (a["C"] @ a["velocity"]).reshape(-1, 3)
        target = bias.reshape(-1, 3)
        loaded = triples[:, 0] > 0
        interior = loaded & (np.linalg.norm(triples[:, 1:], axis=1) < a["mu"] * triples[:, 0]) & (a["mu"] > 0)
        residual.update(
            base_velocity=d["velocity"][1].astype(float).tolist(),
            base_angular_velocity=d["angular_velocity"][1].astype(float).tolist(),
            contacts=len(triples),
            loaded=int(loaded.sum()),
            interior=int(interior.sum()),
            normal_impulse_sum=float(triples[:, 0].sum()),
            tangent_target_max=float(np.max(abs(target[:, 1:]))),
            tangent_velocity_max=float(np.max(abs(jv[:, 1:]))),
            interior_tangent_residual_max=float(np.max(abs((jv + target)[interior, 1:]))) if interior.any() else None,
        )
        result["phases"][label] = residual
    before = z["integrate_before.position"].astype(float)
    after = z["integrate_after.position"].astype(float)
    result["last_integration_base_delta_m"] = (after[1] - before[1]).tolist()
    result["last_integration_base_rounding_m"] = (
        after[1] - before[1] - float(z["dt"][0]) * z["integrate_before.velocity"][1]
    ).tolist()
    path = args.snapshot.with_suffix(".audit.json")
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
