"""Bounded physical-vector normal proposals; frozen safeguards stay unchanged."""

import json
import time
import types
from pathlib import Path

import numpy as np

from local_studies.colibri import physical_face_active_set as frozen
from local_studies.colibri.coupled_support_online import assemble_snapshot
from local_studies.colibri.physical_face_support import factor


def normal_proposal(a, seed, f, sweeps):
    """Update one seven-component response, never a contact-by-contact matrix."""
    trial = seed.copy()
    u = f.T @ trial
    normals = f[::3]
    diagonal = np.sum(normals * normals, axis=1) + a["gamma"]
    for _ in range(sweeps):
        for p in range(len(a["mu"])):
            row = 3 * p
            gradient = normals[p] @ u + a["rhs"][row] + a["gamma"][p] * trial[row]
            normal = max(0.0, trial[row] - gradient / diagonal[p])
            u += normals[p] * (normal - trial[row])
            trial[row] = normal
    gradient = normals @ u + a["rhs"][::3] + a["gamma"] * trial[::3]
    natural = diagonal * (trial[::3] - np.maximum(0.0, trial[::3] - gradient / diagonal))
    for p in range(len(a["mu"])):
        tangent = trial[3 * p + 1 : 3 * p + 3]
        radius = a["mu"][p] * trial[3 * p]
        length = np.linalg.norm(tangent)
        if length > radius:
            tangent *= radius / length
    return trial, {
        "iterations": sweeps,
        "status": "Bounded normal-only SOR1 sweeps, original-order physical-vector updates",
        "normal_error": float(np.max(abs(natural))),
        "normal_rows": len(a["mu"]),
        "physical_dimension": f.shape[1],
        "contact_squared_storage": False,
    }


def solve(a, sweeps=8):
    """Reuse precisely the frozen face/merit gates without mutating its module."""
    f, _ = factor(a)

    def proposal(operator, seed):
        return normal_proposal(operator, seed, f, sweeps)

    namespace = dict(frozen.solve.__globals__)
    namespace["normal_proposal"] = proposal
    function = types.FunctionType(frozen.solve.__code__, namespace)
    value, report = function(a)
    report["bounds"].pop("normal_qp_iterations")
    report["bounds"]["normal_proposal_sweeps"] = sweeps
    report["scope"] += "; SLSQP normal proposal replaced only; source function unchanged"
    return value, report


def snapshots():
    """Use the same three operators as the independent neighborhood study."""
    result = {}
    with np.load("/tmp/colibri_base_frame_totalnormal_phases330.npz") as saved:
        for phase in ("biased", "relax"):
            fields = {k.split(".", 1)[1]: saved[k] for k in saved.files if k.startswith(phase + "_solved.")}
            result[phase] = assemble_snapshot(fields, phase, float(saved["dt"][0]), int(saved["num_joints"][0]))
    with np.load("/tmp/colibri_two_body_condensed_combined_live60.rejected.npz") as saved:
        result["transition"] = dict(saved)
    return result


def main():
    """Reproduce the exact seed4107 physical perturbations for both budgets."""
    records = []
    original_cases = snapshots()
    for sweeps in (8, 16):
        random = np.random.default_rng(4107)
        for name, original in original_cases.items():
            direction = random.normal(size=12)
            direction /= np.linalg.norm(direction)
            for amplitude in (0.0, 1e-6, -1e-6, 1e-4, -1e-4, 1e-3):
                current = dict(original)
                current["free"] = original["free"] + amplitude * direction
                current["velocity"] = original["velocity"] + amplitude * direction
                current["vbar"] = current["free"] + original["W"] @ original["B"].T @ np.linalg.solve(
                    original["K"], original["targets"] - original["B"] @ current["free"]
                )
                current["rhs"] = original["rhs"] + original["C"] @ (current["vbar"] - original["vbar"])
                started = time.perf_counter()
                try:
                    _, report = solve(current, sweeps)
                except (ValueError, AssertionError, np.linalg.LinAlgError) as error:
                    report = {"audit": {"accepted": False}, "failure": repr(error)}
                records.append(
                    {
                        "sweeps": sweeps,
                        "snapshot": name,
                        "perturbation": amplitude,
                        "seconds_cpu_reference": time.perf_counter() - started,
                        **report,
                    }
                )
                print(sweeps, name, amplitude, report["audit"], flush=True)
    Path("/tmp/colibri_physical_face_normal_pgs.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
