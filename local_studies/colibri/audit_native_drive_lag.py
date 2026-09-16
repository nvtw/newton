"""Separate frozen native drive target lag from constitutive equation error."""

import json
from pathlib import Path

import numpy as np

from newton._src.solvers.phoenx.constraints import constraint_joint as schema


def main():
    d = np.load("/tmp/colibri_support_relax330.npz")
    jd = d["joint_data"].view(np.int32)
    result = []
    for joint, count in enumerate(d["joint_row_count"]):
        a, b = jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], joint]
        structural = int(d["joint_structural_index"][joint])
        for row in d["joint_row_indices"][joint, :count]:
            if not d["joint_row_dynamic"][row]:
                continue
            local = d["joint_row_local"][row]
            ja, jb = d["joint_wrench0"][structural, local], d["joint_wrench1"][structural, local]
            for stage, prefix in (("before_relax", ""), ("after_relax", "after_")):
                v = np.concatenate((d[prefix + "velocity"], d[prefix + "angular_velocity"]), axis=1).astype(float)
                speed = float(ja @ v[a] + jb @ v[b])
                reference = float(d["joint_reference"][row])
                impulse = float(d[prefix + "joint_accumulated"][row])
                compliance = 1.0 / float(d["joint_dynamic_mass"][row])
                result.append(
                    {
                        "joint": joint,
                        "row": int(row),
                        "bodies": [int(a), int(b)],
                        "stage": stage,
                        "speed": speed,
                        "reference": reference,
                        "target_error": speed - reference,
                        "generalized_impulse": impulse,
                        "impulse_units": "N m s",
                        "compliance": compliance,
                        "compliance_units": "rad / (N m s^2)",
                        "compliant_load_lag": -compliance * impulse,
                        "constitutive_residual": speed - reference + compliance * impulse,
                    }
                )
    Path("/tmp/colibri_native_drive_lag.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
