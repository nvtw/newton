"""Check physical drive residuals against their final color-group copies.

CPU-only saved-state diagnostic; this does not execute a solver or alter physics.
"""

import json
from pathlib import Path

import numpy as np

from newton._src.solvers.phoenx.constraints import constraint_joint as schema


def main():
    d = np.load("/tmp/colibri_support_relax330.npz")
    c = np.load("/tmp/colibri_ground_pre_average330_copies.npz")
    jd = d["joint_data"].view(np.int32)
    used = [set() for _ in d["position"]]
    colors = []
    for joint in range(int(d["num_joints"][0])):
        endpoints = jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], joint]
        occupied = set().union(*(used[int(b)] for b in endpoints if b >= 0))
        color = 0
        while color in occupied:
            color += 1
        colors.append(color)
        for body in endpoints:
            if body >= 0:
                used[int(body)].add(color)
    assert colors[:2] == [0, 1]
    result = []
    physical = np.concatenate((d["after_velocity"], d["after_angular_velocity"]), axis=1).astype(float)
    copies = np.concatenate((c["velocity"], c["angular_velocity"]), axis=1).astype(float)
    for joint, count in enumerate(d["joint_row_count"]):
        endpoints = jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], joint]
        structural = int(d["joint_structural_index"][joint])
        partition = colors[joint] // 4
        slots = []
        for body in endpoints:
            end = int(c["section_end"][body])
            start = 0 if body == 0 else int(c["section_end"][body - 1])
            found = np.flatnonzero(c["partition_list"][start:end] == partition)
            assert len(found) == 1
            slots.append(start + int(found[0]))
        for row in d["joint_row_indices"][joint, :count]:
            if not d["joint_row_dynamic"][row]:
                continue
            local = d["joint_row_local"][row]
            wrenches = (d["joint_wrench0"][structural, local], d["joint_wrench1"][structural, local])
            correction = float(d["after_joint_accumulated"][row]) / float(d["joint_dynamic_mass"][row])
            reference = float(d["joint_reference"][row])
            physical_residual = (
                sum(float(w @ physical[b]) for w, b in zip(wrenches, endpoints, strict=True)) + correction - reference
            )
            copy_residual = (
                sum(float(w @ copies[s]) for w, s in zip(wrenches, slots, strict=True)) + correction - reference
            )
            result.append(
                {
                    "joint": joint,
                    "row": int(row),
                    "color": colors[joint],
                    "partition": partition,
                    "copy_residual": copy_residual,
                    "physical_residual": physical_residual,
                    "averaging_change": physical_residual - copy_residual,
                }
            )
    output = "/tmp/colibri_drive_copy_accounting_independent.json"
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
