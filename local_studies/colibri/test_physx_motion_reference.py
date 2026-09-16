# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent closed-form CPU checks for the all-FP32 GPU TGS motion reference."""

import json
from pathlib import Path

import numpy as np

from .physx_motion_reference import advance_motion, contact_error, start_motion, writeback_pose


def main():
    """Check stationary, translating, rotating, scaled-coordinate and velocity phases."""
    zero = np.zeros(3, np.float32)
    identity = np.eye(3, dtype=np.float32)
    h = np.float32(1 / 3600)
    q0 = np.array([0, 0, 0, 1], np.float32)
    p0 = np.array([10, -0.02, 0.003], np.float32)
    still = start_motion()
    for _ in range(30):
        advance_motion(still, zero, zero, identity, h)
    np.testing.assert_array_equal(still.linear_delta, zero)
    np.testing.assert_array_equal(still.angular_delta, zero)
    np.testing.assert_array_equal(still.rotation_delta, q0)
    p, q = writeback_pose(p0, q0, still)
    np.testing.assert_array_equal(p, p0)
    np.testing.assert_array_equal(q, q0)

    velocity = np.array([1e-4, -2e-5, 3e-5], np.float32)
    translating = start_motion()
    naive_position = p0.copy()
    for _ in range(30):
        advance_motion(translating, velocity, zero, identity, h)
        naive_position += velocity * h
    ideal_delta = velocity.astype(float) * float(h) * 30
    np.testing.assert_allclose(translating.linear_delta, ideal_delta, atol=3e-13, rtol=1e-6)
    np.testing.assert_array_equal(translating.linear_delta, translating.position_delta)
    p, _ = writeback_pose(p0, q0, translating)
    assert p[0] > p0[0] and naive_position[0] == p0[0]
    # Source velocity pass is a no-op for all motion state, even for large v.
    before = [a.copy() for a in vars(translating).values()]
    advance_motion(
        translating, velocity * np.float32(100), np.ones(3, np.float32), identity, h, velocity_iteration=True
    )
    for actual, expected in zip(vars(translating).values(), before, strict=True):
        np.testing.assert_array_equal(actual, expected)

    omega = np.array([0, 0, 3.6], np.float32)
    rotating = start_motion()
    for _ in range(30):
        advance_motion(rotating, zero, omega, identity, h)
    theta = float(omega[2]) * float(h) * 30
    expected_q = np.array([0, 0, np.sin(theta / 2), np.cos(theta / 2)])
    np.testing.assert_allclose(rotating.rotation_delta, expected_q, rtol=0, atol=2e-7)
    np.testing.assert_allclose(rotating.angular_delta, omega.astype(float) * float(h) * 30, rtol=1e-6, atol=1e-9)
    tangent = np.array([1, 0, 0], np.float32)
    lever = np.array([0.13, 0.1, 0], np.float32)
    row = np.cross(lever, tangent)
    predicted = contact_error(np.float32(0), tangent, row, zero, rotating, still, np.float32(0), np.float32(0))
    np.testing.assert_allclose(predicted, -float(lever[1]) * theta, rtol=1e-6, atol=1e-9)
    exact_finite_motion = float(lever[0]) * (np.cos(theta) - 1) - float(lever[1]) * np.sin(theta)
    # This difference is intended source linearization, not a test tolerance excuse.
    assert abs(float(predicted) - exact_finite_motion) > 5e-5

    factor = np.diag(np.array([2, 3, 4], np.float32))
    u = np.array([0.1, -0.2, 0.3], np.float32)
    scaled = start_motion()
    for _ in range(30):
        advance_motion(scaled, velocity, u, factor, h)
    scaled_error = contact_error(
        np.float32(0.002), tangent, factor @ row, zero, scaled, still, np.float32(0.01), np.float32(30) * h
    )
    world_error = (
        np.float32(0.002)
        - np.float32(0.01) * np.float32(30) * h
        + np.dot(row, factor @ scaled.angular_delta)
        + np.dot(tangent, scaled.linear_delta)
    )
    np.testing.assert_allclose(scaled_error, world_error, rtol=0, atol=2e-9)
    common_translation_error = contact_error(
        np.float32(0), tangent, zero, zero, translating, translating, np.float32(0), np.float32(0)
    )
    assert common_translation_error == 0
    report = {
        "candidate_dtype": "FP32 only; independent analytic oracle uses FP64",
        "stationary_exact": True,
        "velocity_pass_does_not_integrate": True,
        "translation_delta_error_m": float(np.max(abs(translating.linear_delta - ideal_delta))),
        "deferred_world_x_change_m": float(p[0] - p0[0]),
        "per_microstep_world_x_change_m": float(naive_position[0] - p0[0]),
        "rotation_quaternion_error": float(np.max(abs(rotating.rotation_delta - expected_q))),
        "contact_linearized_rotation_error_m": float(predicted),
        "finite_rotation_witness_error_m": exact_finite_motion,
        "scaled_vs_world_coordinate_error_m": float(abs(scaled_error - world_error)),
        "scope": "Semantic CPU reference; no GPU bitwise, full-solver, sleep/lock or live-creep claim.",
    }
    Path("/tmp/colibri_physx_motion_reference.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
