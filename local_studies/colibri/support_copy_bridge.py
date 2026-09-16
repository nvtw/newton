"""Frozen copy ownership/averaging audit; never inject into the live solver."""

import argparse
import json
from pathlib import Path

import numpy as np

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.mass_splitting.copy_state import copy_state_container_zeros
from newton._src.solvers.phoenx.mass_splitting.kernels import (
    launch_average_and_broadcast_rigid_velocity,
    launch_copy_state_into_rigids,
)
from newton._src.solvers.phoenx.particle import ParticleContainer


def main(device="cpu", reference="/tmp/colibri_support_copy_reference"):
    """Apply one owned component update, then the real canonical average."""
    d = np.load("/tmp/colibri_support_relax330.npz")
    graph = np.load("/tmp/colibri_support_bridge_counts.npz")
    r = np.load(reference + ".npz")
    counts = graph["star"]
    ids = [1, 2]
    partition = int(graph["row_partition"][0])
    memberships = [
        sorted(
            {
                int(pid)
                for row, pid in zip(
                    graph["elements"]["bodies"], graph["row_partition"][: len(graph["elements"])], strict=True
                )
                if b in row
            }
        )
        for b in range(len(counts))
    ]
    np.testing.assert_array_equal([len(x) for x in memberships], counts)
    ends = np.cumsum(counts, dtype=np.int32)
    starts = ends - counts
    slots = [int(starts[b]) + memberships[b].index(partition) for b in ids]
    cs = copy_state_container_zeros(int(ends[-1]), len(counts), device)
    bodies = body_container_zeros(len(counts), device)
    particles = ParticleContainer()
    cs.section_end.assign(ends)
    cs.count_per_node.assign(counts)
    cs.partition_list.assign(np.concatenate(memberships).astype(np.int32))
    cs.highest_index_in_use.assign(np.array([ends[-1]], np.int32))
    cs.access_mode.fill_(ACCESS_MODE_VELOCITY_LEVEL)
    for field in ("position", "orientation", "velocity", "angular_velocity"):
        getattr(bodies, field).assign(d[field])
        getattr(cs, field).assign(np.repeat(d[field], counts, axis=0))
    # Only these two owned copies receive the component's copy-mass response.
    velocity = cs.velocity.numpy()
    angular = cs.angular_velocity.numpy()
    for i, slot in enumerate(slots):
        velocity[slot] = r["v"].reshape(-1, 6)[i, :3]
        angular[slot] = r["v"].reshape(-1, 6)[i, 3:]
    cs.velocity.assign(velocity)
    cs.angular_velocity.assign(angular)
    launch_average_and_broadcast_rigid_velocity(cs, bodies, particles, len(counts), 3600.0)
    launch_copy_state_into_rigids(cs, bodies, particles, len(counts), 3600.0)
    actual = (
        np.concatenate((bodies.velocity.numpy()[ids], bodies.angular_velocity.numpy()[ids]), axis=1)
        .astype(float)
        .ravel()
    )
    u = r["u"]
    f = np.linalg.solve(r["W"], r["v"] - u)
    expected = u + r["Wphysical"] @ f
    M = np.linalg.inv(r["Wphysical"])
    selected = (3 * r["active"][:, None] + np.arange(3)).ravel()
    C = r["J"][selected]
    contact = C.T @ (r["solution"] - r["initial"])
    hard = r["Jb"].T @ r["beta"]
    drive = r["Js"].T @ r["delta_soft"]
    np.testing.assert_allclose(f, contact + hard + drive, atol=1e-12, rtol=1e-10)
    average_loss = 0.5 * f @ (r["W"] - r["Wphysical"]) @ f
    copy_delta = 0.5 * (r["v"] @ np.linalg.solve(r["W"], r["v"]) - u @ np.linalg.solve(r["W"], u))
    delta = 0.5 * (expected @ M @ expected - u @ M @ u)
    actual_delta = 0.5 * (actual @ M @ actual - u @ M @ u)
    momentum_error = np.zeros(6)
    for i, body in enumerate(ids):
        sl = slice(6 * i, 6 * i + 6)
        mismatch = (M @ (actual - u) - contact)[sl]
        momentum_error[:3] += mismatch[:3]
        momentum_error[3:] += mismatch[3:] + np.cross(d["position"][body], mismatch[:3])
    midpoint = (u + expected) * 0.5
    # Restore only the native normal compliance/bias offset; geometry is unchanged.
    offset = r["A"] @ r["solution"] + r["rhs"] - C @ r["v"]
    gradient = C @ actual + offset
    gradient[::3] += r["regularization"] * r["solution"][::3]
    triples = r["solution"].reshape(-1, 3)
    grad = gradient.reshape(-1, 3)
    scale = np.diag(r["A"]).reshape(-1, 3).max(axis=1)
    normal = max(np.max(np.maximum(-grad[:, 0], 0)), np.max(np.abs(np.minimum(triples[:, 0] * scale, grad[:, 0]))))
    tangent = []
    for value, g, mobility in zip(triples, grad, scale, strict=True):
        trial = value[1:] - g[1:] / mobility
        projected = trial * min(1.0, 0.5 * max(value[0], 0) / max(np.linalg.norm(trial), 1e-300))
        tangent.append(np.linalg.norm((value[1:] - projected) * mobility))
    report = {
        "device": device,
        "stage": "Frozen broadcast -> replace joint0 and support rows once -> canonical average/writeback",
        "outside_rows": "Held fixed; no native solve is run before or after the owned component",
        "component_partition": partition,
        "owned_slots": slots,
        "copy_counts": counts[ids].tolist(),
        "copy_reference_accepted": json.loads(Path(reference + ".json").read_text())["accepted"],
        "canonical_average_error": float(np.max(np.abs(actual - expected))),
        "physical_momentum_minus_static_reaction": momentum_error.tolist(),
        "averaged_normal_residual": float(normal),
        "averaged_tangent_residual": float(max(tangent)),
        "averaged_drive_equation_residual": float(
            np.max(np.abs(r["Js"] @ actual + r["Ds"] * (r["old_soft"] + r["delta_soft"]) - r["reference_soft"]))
        ),
        "averaged_joint_row_residuals": (r["Jb"] @ actual).tolist(),
        "joint_row_linear_norms": np.linalg.norm(r["Jb"].reshape(-1, 2, 6)[:, :, :3], axis=(1, 2)).tolist(),
        "joint_row_angular_norms": np.linalg.norm(r["Jb"].reshape(-1, 2, 6)[:, :, 3:], axis=(1, 2)).tolist(),
        "copy_joint_residual": float(np.max(np.abs(r["Jb"] @ r["v"]))),
        "averaged_joint_residual": float(np.max(np.abs(r["Jb"] @ actual))),
        "copy_energy_change_J": float(copy_delta),
        "averaging_dissipation_J": float(average_loss),
        "physical_energy_change_J": float(delta),
        "canonical_physical_energy_change_J": float(actual_delta),
        "copy_minus_averaging_balance_J": float(copy_delta - average_loss - delta),
        "physical_contact_work_J": float(contact @ midpoint),
        "physical_hard_joint_work_J": float(hard @ midpoint),
        "physical_drive_work_J": float(drive @ midpoint),
        "physical_work_balance_J": float(delta - (contact + hard + drive) @ midpoint),
        "live_candidate_accepted": False,
    }
    prefix = "/tmp/colibri_support_copy_bridge_" + device.replace(":", "_")
    Path(prefix + ".json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(prefix + ".npz", expected=expected, actual=actual, impulse=f, counts=counts, slots=slots)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference", default="/tmp/colibri_support_copy_reference")
    args = parser.parse_args()
    main(args.device, args.reference)
