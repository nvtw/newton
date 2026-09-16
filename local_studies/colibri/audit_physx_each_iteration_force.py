"""Audit the single-setting PhysX external-force timing companion."""

import hashlib
import json
from pathlib import Path

import numpy as np

from local_studies.colibri.audit_native_conditioned_motion import relative


def main():
    """Compare identical runtime properties and common trajectory windows."""
    baseline = Path("/tmp/colibri_physx_two_body_awake_plane1mm60")
    candidate = Path("/tmp/colibri_physx_two_body_awake_plane1mm_eachforce60")
    b = np.load(str(baseline) + ".npz")
    c = np.load(str(candidate) + ".npz")
    assert set(b.files) == set(c.files)
    properties = {}
    for key in b.files:
        if key not in ("q", "qd"):
            assert b[key].dtype == c[key].dtype and b[key].tobytes() == c[key].tobytes(), key
            properties[key] = True
    results = {}
    for path, data in ((baseline, b), (candidate, c)):
        q = data["q"]
        assert q.shape == (3601, 2, 7) and np.isfinite(q).all() and np.isfinite(data["qd"]).all()
        windows = {}
        for lo, hi in ((5, 10), (10, 60), (30, 60), (50, 60)):
            start, end = 60 * lo, 60 * hi
            delta = q[end, 0, :3] - q[start, 0, :3]
            windows[f"{lo}-{hi}"] = {
                "xy_delta_um": (delta[:2] * 1e6).tolist(),
                "net_xy_um": float(np.linalg.norm(delta[:2]) * 1e6),
                "rate_um_s": float(np.linalg.norm(delta[:2]) * 1e6 / (hi - lo)),
                "max_excursion_um": float(
                    np.max(np.linalg.norm(q[start : end + 1, 0, :2] - q[start, 0, :2], axis=1)) * 1e6
                ),
            }
        hinge = {}
        for seconds in (10, 60):
            r = relative(q[seconds * 60, 0, 3:], q[seconds * 60, 1, 3:])
            hinge[str(seconds)] = float(np.degrees(2 * np.arctan2(r[2], r[3])))
        results[path.name] = {"windows": windows, "hinge_deg": hinge}
    fixture = json.loads(Path("/tmp/colibri_physx_two_body_awake_plane1mm_eachforce.fixture.json").read_text())
    assert len(fixture["composed_differences"]) == 1
    for role in ("base", "overlay"):
        assert hashlib.sha256(Path(fixture[role]).read_bytes()).hexdigest() == fixture[role + "_sha256"]
    source_result = json.loads(Path(str(candidate) + ".json").read_text())
    assert source_result["source_sha256"] == fixture["overlay_sha256"]
    report = {
        "runtime_properties_byte_identical": properties,
        "fixture": fixture,
        "results": results,
        "environment": "Logs for both runs resolve ovphysx/ovstage in uv archive j4xPZTZuhlAwReXiGUPET; ovphysx0.5.11",
        "scope": "Accuracy trajectories; no performance claim. Attribute/importer verified; no runtime scene-flag getter exposed.",
    }
    Path("/tmp/colibri_physx_eachforce_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
