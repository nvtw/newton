"""Compare native final-relax counts at one identical public Colibri state."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints import constraint_joint as schema


def residuals(snapshot, velocity, angular, impulses, accumulated):
    """Measure physical contact KKT and native joint constitutive residuals."""
    s = snapshot
    v = np.concatenate((velocity, angular), axis=1).astype(float)
    h = s["headers"].view(np.int32)
    normal, tangent = [], []
    W = np.zeros((len(s["inverse_mass"]), 6, 6))
    for body, inverse_mass in enumerate(s["inverse_mass"]):
        W[body, :3, :3] = np.eye(3) * float(inverse_mass)
        xx, yy, zz, xy, xz, yz = s["inverse_inertia"][body].astype(float)
        W[body, 3:, 3:] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    for col in range(int(s["column_count"][0])):
        a, b = h[1:3, col]
        for point in range(h[5, col], h[5, col] + h[6, col]):
            if s["derived"][3, point] > 0:
                continue
            n = s["lambdas"][:3, point].astype(float)
            t = s["lambdas"][3:6, point].astype(float)
            axes = np.array([n, t, np.cross(n, t)])
            ca = -np.concatenate((axes, np.cross(s["derived"][9:12, point].astype(float), axes)), axis=1)
            cb = np.concatenate((axes, np.cross(s["derived"][12:15, point].astype(float), axes)), axis=1)
            g = ca @ v[a] + cb @ v[b]
            H = ca @ W[a] @ ca.T + cb @ W[b] @ cb.T
            scale = float(np.diag(H).max())
            lam = impulses[:, point].astype(float)
            trial = lam[1:] - g[1:] / scale
            radius = float(s["headers"][4, col]) * max(lam[0], 0)
            proj = trial * min(1, radius / max(np.linalg.norm(trial), 1e-300))
            normal.append(max(-g[0], -lam[0] * scale, abs(min(lam[0] * scale, g[0])), 0))
            tangent.append(float(np.linalg.norm(lam[1:] - proj) * scale))
    jd = s["joint_data"].view(np.int32)
    joints, drives = [], []
    for joint, count in enumerate(s["joint_row_count"]):
        a, b = jd[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], joint]
        structural = int(s["joint_structural_index"][joint])
        for row in s["joint_row_indices"][joint, :count]:
            local = s["joint_row_local"][row]
            ja, jb = s["joint_wrench0"][structural, local], s["joint_wrench1"][structural, local]
            speed = float(ja @ v[a] + jb @ v[b])
            compliance = 1 / float(s["joint_dynamic_mass"][row]) if s["joint_row_dynamic"][row] else 0
            reference = float(s["joint_reference"][row])
            r = speed + compliance * float(accumulated[row]) - reference
            joints.append(abs(r))
            if s["joint_row_dynamic"][row]:
                drives.append(
                    {
                        "row": int(row),
                        "speed": speed,
                        "reference": reference,
                        "impulse": float(accumulated[row]),
                        "constitutive_residual": r,
                    }
                )
    return {
        "normal_residual": max(normal),
        "tangent_residual": max(tangent),
        "max_joint_residual": max(joints),
        "eligible_points": len(normal),
        "drives": drives,
    }


def validate_reference(actual_path, reference_path, mode):
    """Require complete byte identity for the selected independent reference."""
    actual_path, reference_path = Path(actual_path), Path(reference_path)
    if actual_path.resolve() == reference_path.resolve():
        raise ValueError("Capture output cannot overwrite its independent reference")
    required = ("q", "qd", "initial_q", "q_history", "qd_history", "history_times", "labels")
    with np.load(actual_path) as actual, np.load(reference_path) as reference:
        keys = list(reference.files)
        if mode == "trajectory":
            missing = [key for key in required if key not in reference.files]
            if missing:
                raise AssertionError(f"Trajectory reference lacks required history fields: {missing}")
        elif mode != "snapshot":
            raise ValueError(f"Unknown reference mode: {mode}")
        if not keys:
            raise AssertionError("Independent reference archive is empty")
        for key in keys:
            if key not in actual.files:
                raise AssertionError(f"Capture lacks reference field: {key}")
            a, b = actual[key], reference[key]
            if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                raise AssertionError(f"{mode} reference mismatch: {key}")
    return {
        "mode": mode,
        "reference_path": str(reference_path),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "matched_keys": keys,
        "byte_identical": True,
    }


def main():
    """Capture current330 once, then replay independent native relaxation counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/colibri_native_velocity_iterations")
    references = parser.add_mutually_exclusive_group()
    references.add_argument(
        "--reference-trajectory",
        type=Path,
        help="Require current capture q/qd and complete history to match this independent trajectory",
    )
    references.add_argument(
        "--reference-snapshot", type=Path, help="Require all fields of an explicit independent phase snapshot to match"
    )
    args = parser.parse_args()
    reference_mode = "trajectory" if args.reference_trajectory is not None else "snapshot"
    reference_path = args.reference_trajectory or args.reference_snapshot or Path("/tmp/colibri_support_relax330.npz")
    if not reference_path.is_file():
        parser.error(f"Independent reference does not exist: {reference_path}")
    reference_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    outputs = [Path(args.output + suffix).resolve() for suffix in ("_snapshot.npz", "_trajectory.npz", ".npz")]
    if reference_path.resolve() in outputs:
        parser.error("Output would overwrite the independent reference")

    prefix = args.output
    sys.argv = [
        sys.argv[0],
        "--frames",
        "330",
        "--save-history",
        "--snapshot",
        prefix + "_snapshot.npz",
        "--output",
        prefix + "_trajectory.json",
    ]
    capture = runpy.run_module("local_studies.colibri.capture_support_relax", run_name="__main__")
    state = capture["captured"]
    w = state["world"]
    before = state["before"]
    destinations = {
        "velocity": w.bodies.velocity,
        "angular_velocity": w.bodies.angular_velocity,
        "impulses": w._contact_container.impulses,
        "joint_accumulated": w.constraints.bilateral.accumulated,
        "lambdas": w._contact_container.lambdas,
        "derived": w._contact_container.derived,
    }
    s = np.load(prefix + "_snapshot.npz")
    validation_path = prefix + ("_trajectory.npz" if reference_mode == "trajectory" else "_snapshot.npz")
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != reference_digest:
        raise AssertionError("Independent reference changed during capture")
    validation = validate_reference(validation_path, reference_path, reference_mode)
    original_count = w.velocity_iterations
    reports = []
    results = {}
    for count in (1, 2, 4):
        for name, array in destinations.items():
            wp.copy(array, before[name])
        w.velocity_iterations = count
        assert w._active_velocity_iterations == count
        w._dispatcher.relax(wp.float32(1 / w.substep_dt))
        wp.synchronize_device(w.bodies.velocity.device)
        out = {name: array.numpy() for name, array in destinations.items()}
        if count == 1:
            for name in ("velocity", "angular_velocity", "impulses", "joint_accumulated"):
                assert out[name].tobytes() == s["after_" + name].tobytes(), name
        results.update({str(count) + "_" + k: v for k, v in out.items()})
        report = residuals(s, out["velocity"], out["angular_velocity"], out["impulses"], out["joint_accumulated"])
        report["velocity_iterations"] = count
        reports.append(report)
        print("FROZEN_NATIVE_RELAX", json.dumps(report), flush=True)
    w.velocity_iterations = original_count
    report = {
        "reference_validation": validation,
        "source_snapshot_byte_identical": reference_mode == "snapshot",
        "source_trajectory_byte_identical": reference_mode == "trajectory",
        "physical_operator_source": prefix + "_snapshot.npz",
        "one_sweep_byte_identical": True,
        "substeps": 30,
        "collision_hz": 120,
        "velocity_relaxation": "final_substep",
        "scope": "Frozen final330 pre-relax state; increased native work, not a matched velocity budget",
        "reports": reports,
    }
    Path(prefix + ".json").write_text(json.dumps(report, indent=2))
    np.savez(prefix + ".npz", **results)


if __name__ == "__main__":
    main()
