"""Validate a covariant high/low material-anchor representation in isolation."""

import json
from pathlib import Path

import numpy as np


def main():
    """Check birth cancellation and physical motion without rounding away offsets."""
    from scipy.spatial.transform import Rotation

    random = np.random.default_rng(7503)
    metrics = {"birth_error_m": 0.0, "motion_error_m": 0.0, "covariance_error_m": 0.0, "uncorrected_birth_error_m": 0.0}
    for _ in range(200):
        positions = random.normal(size=(2, 3))
        rotations = Rotation.random(2, random_state=random).as_matrix()
        point = random.normal(size=3)
        anchors = np.asarray([rotations[i].T @ (point - positions[i]) for i in range(2)], dtype=np.float32)
        # Preserve the existing coarse anchors and retain the tiny residual as
        # a separate float32 vector. Evaluate all transforms in float64.
        birth_difference = positions[1] + rotations[1] @ anchors[1].astype(float)
        birth_difference -= positions[0] + rotations[0] @ anchors[0].astype(float)
        correction = (rotations[0].T @ birth_difference).astype(np.float32)
        corrected_birth = birth_difference - rotations[0] @ correction.astype(float)
        metrics["uncorrected_birth_error_m"] = max(
            metrics["uncorrected_birth_error_m"], float(np.max(abs(birth_difference)))
        )
        metrics["birth_error_m"] = max(metrics["birth_error_m"], float(np.max(abs(corrected_birth))))
        p = positions + random.normal(size=(2, 3)) * 1e-4
        r = Rotation.from_rotvec(random.normal(size=(2, 3)) * 1e-4).as_matrix() @ rotations
        delta = p[1] + r[1] @ anchors[1].astype(float) - p[0] - r[0] @ anchors[0].astype(float)
        delta -= r[0] @ correction.astype(float)
        corrected_anchor0 = anchors[0].astype(float) + correction.astype(float)
        expected = p[1] + r[1] @ anchors[1].astype(float) - p[0] - r[0] @ corrected_anchor0
        metrics["motion_error_m"] = max(metrics["motion_error_m"], float(np.max(abs(delta - expected))))
        rotation = Rotation.random(random_state=random).as_matrix()
        translation = random.normal(size=3)
        transformed_p = p @ rotation.T + translation
        transformed_r = rotation @ r
        transformed_delta = transformed_p[1] + transformed_r[1] @ anchors[1].astype(float)
        transformed_delta -= transformed_p[0] + transformed_r[0] @ anchors[0].astype(float)
        transformed_delta -= transformed_r[0] @ correction.astype(float)
        metrics["covariance_error_m"] = max(
            metrics["covariance_error_m"], float(np.max(abs(transformed_delta - rotation @ delta)))
        )
    assert metrics["birth_error_m"] < 1e-12
    assert metrics["motion_error_m"] < 1e-12
    assert metrics["covariance_error_m"] < 1e-12
    report = {
        "cases": 200,
        **metrics,
        "scope": "Independent anchor representation only; three float32 correction components, float64 transforms; no live solver claim",
    }
    Path("/tmp/colibri_birth_residual_reference.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
