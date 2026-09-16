"""Check a relative-pose material reference independently of solver kernels."""

import json
from pathlib import Path

import numpy as np


def relative_pose(p0, r0, p1, r1):
    """Map COM-relative material coordinates from body one into body zero."""
    return r0.T @ (p1 - p0), r0.T @ r1


def displacement(reference, p0, r0, p1, r1, anchor1):
    """Evaluate material displacement from changes in the relative pose."""
    translation, rotation = relative_pose(p0, r0, p1, r1)
    translation_birth, rotation_birth = reference
    return r0 @ ((translation - translation_birth) + (rotation - rotation_birth) @ anchor1)


def main():
    """Verify zero birth error, material motion and rigid-frame covariance."""
    from scipy.spatial.transform import Rotation

    random = np.random.default_rng(9432)
    maximum_material_error = 0.0
    maximum_covariance_error = 0.0
    maximum_single_covariance_error = 0.0
    for _ in range(200):
        positions = random.normal(size=(2, 3))
        rotations = Rotation.random(2, random_state=random).as_matrix()
        anchor1 = random.normal(size=3) * 0.1
        reference = relative_pose(positions[0], rotations[0], positions[1], rotations[1])
        # This anchor is implied by the shared birth pose, not an independently
        # rounded inverse transform of the same absolute world point.
        anchor0 = reference[0] + reference[1] @ anchor1
        current_positions = positions + random.normal(size=(2, 3)) * 1e-4
        current_rotations = Rotation.from_rotvec(random.normal(size=(2, 3)) * 1e-4).as_matrix() @ rotations
        args = (current_positions[0], current_rotations[0], current_positions[1], current_rotations[1], anchor1)
        delta = displacement(reference, *args)
        material_delta = current_positions[1] + current_rotations[1] @ anchor1
        material_delta -= current_positions[0] + current_rotations[0] @ anchor0
        maximum_material_error = max(maximum_material_error, float(np.max(np.abs(delta - material_delta))))

        global_rotation = Rotation.random(random_state=random).as_matrix()
        global_translation = random.normal(size=3)
        transformed_birth_positions = positions @ global_rotation.T + global_translation
        transformed_birth_rotations = global_rotation @ rotations
        transformed_positions = current_positions @ global_rotation.T + global_translation
        transformed_rotations = global_rotation @ current_rotations
        for dtype in (np.float64, np.float32):
            p, r = positions.astype(dtype), rotations.astype(dtype)
            ref = relative_pose(p[0], r[0], p[1], r[1])
            birth_error = displacement(ref, p[0], r[0], p[1], r[1], anchor1.astype(dtype))
            np.testing.assert_array_equal(birth_error, np.zeros(3, dtype=dtype))

            bp, br = transformed_birth_positions.astype(dtype), transformed_birth_rotations.astype(dtype)
            cp, cr = transformed_positions.astype(dtype), transformed_rotations.astype(dtype)
            transformed_reference = relative_pose(bp[0], br[0], bp[1], br[1])
            rotated_delta = displacement(transformed_reference, cp[0], cr[0], cp[1], cr[1], anchor1.astype(dtype))
            error = float(np.max(np.abs(rotated_delta - global_rotation @ delta)))
            if dtype == np.float64:
                maximum_covariance_error = max(maximum_covariance_error, error)
            else:
                maximum_single_covariance_error = max(maximum_single_covariance_error, error)
    assert maximum_material_error < 1e-12
    assert maximum_covariance_error < 1e-12
    report = {
        "cases": 200,
        "birth_error_float32_and_float64": 0.0,
        "maximum_material_motion_error_m": maximum_material_error,
        "maximum_rigid_covariance_error_m": maximum_covariance_error,
        "maximum_float32_covariance_error_m": maximum_single_covariance_error,
        "scope": "Independent algebra only; FP32 finite-motion cancellation remains measured, not solved; no live acceptance",
    }
    Path("/tmp/colibri_material_reference_math.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
