"""Read-only matched-window drift and provenance audit of native sweep counts."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.audit_native_conditioned_motion import angles, relative, world_relative


def main():
    """Compare the same five-to-ten-second interval without changing any check."""
    prefixes = {
        1: "/tmp/colibri_native_direct_owned_fixed3600",
        2: "/tmp/colibri_native_owned_fixed_sweeps2_600",
        4: "/tmp/colibri_native_owned_fixed_sweeps4_600",
    }
    reports = []
    reference = json.loads(Path(prefixes[1] + ".native_conditioned.json").read_text())
    for n, prefix in prefixes.items():
        if not Path(prefix + ".npz").exists():
            continue
        z = np.load(prefix + ".npz")
        t = z["history_times"]
        q = z["q_history"].astype(float)
        lo = int(np.argmin(abs(t - 5)))
        hi = int(np.argmin(abs(t - 10)))
        assert abs(t[lo] - 5) < 1e-8 and abs(t[hi] - 10) < 1e-8
        delta = q[hi, 0, :3] - q[lo, 0, :3]
        provenance = json.load(open(prefix + ".native_conditioned.json"))
        assert provenance["production_unchanged"] and provenance["differential_history_rows"] == 27
        assert provenance["normalized_rotation_increment"]
        assert provenance["actual_owned_modules"] == reference["actual_owned_modules"]
        assert provenance["owned_binding_gate"] == reference["owned_binding_gate"]
        for key, value in reference["source_sha256"].items():
            assert provenance["source_sha256"][key] == value, key
        assert set(provenance["actual_owned_modules"]) == {
            "direct_contact_gs",
            "maximal_contact_gs",
            "reduced_contact_block",
        }
        assert provenance["owned_binding_gate"]["actual_owned_total_normal_load_source"]
        options = provenance["options"]
        assert options["solver_iterations"] == n and options.get("velocity_iterations", 1) == n
        assert options["substeps"] == 30 and options["sor_boost"] == 1
        if n > 1:
            assert provenance["sweep_budget"]["biased_per_frame"] == 60 * n
            assert provenance["sweep_budget"]["unbiased_per_frame"] == 2 * n

        result = json.load(open(prefix + ".json"))
        hinge = relative(q[hi, 0, 3:], q[hi, 1, 3:])
        reports.append(
            {
                "sweeps": n,
                "source": prefix,
                "window": [float(t[lo]), float(t[hi])],
                "delta_um": (delta * 1e6).tolist(),
                "xy_rate_um_s": float(np.linalg.norm(delta[:2]) * 1e6 / (t[hi] - t[lo])),
                "max_excursion_um": float(np.linalg.norm(q[lo : hi + 1, 0, :2] - q[lo, 0, :2], axis=1).max() * 1e6),
                "world_rotation_deg": angles(world_relative(q[lo, 0, 3:], q[hi, 0, 3:])),
                "hinge_deg": float(np.degrees(2 * np.arctan2(hinge[2], hinge[3]))),
                "timing": result["timing"],
                "bounds_passed": result["passed"],
            }
        )
    Path("/tmp/colibri_native_owned_fixed_sweep_comparison.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
