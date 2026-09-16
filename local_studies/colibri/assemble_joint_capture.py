"""Assemble original physical joint equations from an actual relaxation capture."""

import argparse
from pathlib import Path

import numpy as np

from newton._src.solvers.phoenx.constraints import constraint_joint as schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    s = np.load(args.snapshot)
    count = len(s["joint_reference"])
    bodies = len(s["position"])
    J = np.zeros((count, bodies, 6))
    W = np.zeros((bodies, 6, 6))
    header = s["joint_data"].view(np.int32)
    for joint, n in enumerate(s["joint_row_count"]):
        a, b = header[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], joint]
        structural = int(s["joint_structural_index"][joint])
        for row in s["joint_row_indices"][joint, :n]:
            local = s["joint_row_local"][row]
            J[row, a] += s["joint_wrench0"][structural, local]
            J[row, b] += s["joint_wrench1"][structural, local]
    for b in range(bodies):
        W[b, :3, :3] = np.eye(3) * s["inverse_mass"][b]
        xx, yy, zz, xy, xz, yz = s["inverse_inertia"][b].astype(float)
        W[b, 3:, 3:] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    R = np.zeros(count)
    dynamic = (s["joint_row_dynamic"] != 0) & (s["joint_dynamic_mass"] > 0)
    R[dynamic] = 1 / s["joint_dynamic_mass"][dynamic].astype(float)
    initial = np.concatenate((s["after_velocity"], s["after_angular_velocity"]), axis=1).astype(float)
    rhs = np.einsum("rbi,bi->r", J, initial) + R * s["after_joint_accumulated"] - s["joint_reference"]
    A = np.einsum("rbi,bij,sbj->rs", J, W, J) + np.diag(R)
    np.savez(args.output, J=J, W=W, rhs=rhs, initial=initial, compliance=R, A=A)
    print(str(Path(args.output)))


if __name__ == "__main__":
    main()
