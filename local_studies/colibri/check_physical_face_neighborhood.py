"""Probe nearby physical loads without changing contact or drive equations."""

import json
import time
from pathlib import Path

import numpy as np

from local_studies.colibri.coupled_support_online import assemble_snapshot
from local_studies.colibri.physical_face_active_set import solve


def main():
    """Perturb free velocities and independently check complete solve residuals."""
    snapshots = {}
    with np.load("/tmp/colibri_base_frame_totalnormal_phases330.npz") as saved:
        for phase in ("biased", "relax"):
            fields = {key.split(".", 1)[1]: saved[key] for key in saved.files if key.startswith(phase + "_solved.")}
            snapshots[phase] = assemble_snapshot(fields, phase, float(saved["dt"][0]), int(saved["num_joints"][0]))
    with np.load("/tmp/colibri_two_body_condensed_combined_live60.rejected.npz") as saved:
        snapshots["transition"] = {key: saved[key] for key in saved.files}

    random = np.random.default_rng(4107)
    records = []
    for name, original in snapshots.items():
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
                _impulse, result = solve(current)
            except (ValueError, AssertionError, np.linalg.LinAlgError) as error:
                result = {"audit": {"accepted": False}, "failure": repr(error)}
            record = {
                "snapshot": name,
                "free_velocity_perturbation_norm": amplitude,
                "seconds_cpu_reference": time.perf_counter() - started,
                **result,
            }
            records.append(record)
            print("NEIGHBORHOOD", name, amplitude, result["audit"], flush=True)
    output = Path("/tmp/colibri_physical_face_neighborhood.json")
    output.write_text(json.dumps(records, indent=2))
    print("ACCEPTED", sum(record["audit"]["accepted"] for record in records), "OF", len(records), flush=True)


if __name__ == "__main__":
    main()
