"""Check FP32 rotation increments against an offline polynomial reference."""

import json
from pathlib import Path

import numpy as np


def rotate(q, point):
    """Evaluate the usual unit-quaternion rotation polynomial."""
    cross = np.cross(q[:3], point)
    return point + 2 * q[3] * cross + 2 * np.cross(q[:3], cross)


def rotation_increment(q, delta, point):
    """Keep small motion separate instead of subtracting absolute rotations."""
    midpoint = q + np.float32(0.5) * delta
    vm, dv = midpoint[:3], delta[:3]
    return np.float32(2) * (
        delta[3] * np.cross(vm, point)
        + midpoint[3] * np.cross(dv, point)
        + np.cross(dv, np.cross(vm, point))
        + np.cross(vm, np.cross(dv, point))
    )


def main():
    """Measure FP32 error from nanoradian through finite contact motion."""
    random = np.random.default_rng(1088)
    report = []
    for magnitude in (0.0, 1e-9, 1e-7, 1e-5, 1e-3, 1e-2):
        incremental_errors = []
        subtractive_errors = []
        for _ in range(200):
            q = random.normal(size=4)
            q = (q / np.linalg.norm(q)).astype(np.float32)
            delta = (random.normal(size=4) * magnitude).astype(np.float32)
            point = (random.normal(size=3) * 0.1).astype(np.float32)
            actual = rotation_increment(q, delta, point)
            subtractive = rotate(q + delta, point) - rotate(q, point)
            # Higher precision exists only in this independent offline oracle.
            exact = rotate(q.astype(float) + delta.astype(float), point.astype(float))
            exact -= rotate(q.astype(float), point.astype(float))
            incremental_errors.append(float(np.max(abs(actual - exact))))
            subtractive_errors.append(float(np.max(abs(subtractive - exact))))
            if magnitude == 0:
                np.testing.assert_array_equal(actual, np.zeros(3, dtype=np.float32))
        report.append(
            {
                "quaternion_increment_scale": magnitude,
                "max_incremental_error_m": max(incremental_errors),
                "max_subtractive_error_m": max(subtractive_errors),
            }
        )
    Path("/tmp/colibri_quaternion_increment_math.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
