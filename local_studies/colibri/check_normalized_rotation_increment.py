"""Keep quaternion norm changes out of FP32 material rotation increments."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.check_quaternion_increment_math import rotation_increment


def normalized_increment(q, delta, point):
    """Difference normalized rotations using only small FP32 differences."""
    norm_birth = np.dot(q, q)
    norm_delta = np.float32(2) * np.dot(q, delta) + np.dot(delta, delta)
    cross = np.cross(q[:3], point)
    birth_correction = np.float32(2) * (q[3] * cross + np.cross(q[:3], cross))
    numerator = rotation_increment(q, delta, point) - birth_correction * (norm_delta / norm_birth)
    return numerator / (norm_birth + norm_delta)


def normalized_rotate(q, point):
    """Evaluate the independently normalized polynomial for the offline oracle."""
    cross = np.cross(q[:3], point)
    return point + (2 / np.dot(q, q)) * (q[3] * cross + np.cross(q[:3], cross))


def main():
    """Measure radial-noise rejection and small physical rotation accuracy."""
    random = np.random.default_rng(3107)
    reports = []
    for scale in (0.0, 1e-9, 1e-7, 1e-5, 1e-3):
        for mode in ("arbitrary", "radial"):
            maximum = 0.0
            raw_maximum = 0.0
            for _ in range(200):
                q = random.normal(size=4)
                q = (q / np.linalg.norm(q)).astype(np.float32)
                delta = (q * scale if mode == "radial" else random.normal(size=4) * scale).astype(np.float32)
                point = (random.normal(size=3) * 0.1).astype(np.float32)
                actual = normalized_increment(q, delta, point)
                assert actual.dtype == np.float32
                reference = normalized_rotate(q.astype(float) + delta.astype(float), point.astype(float))
                reference -= normalized_rotate(q.astype(float), point.astype(float))
                maximum = max(maximum, float(np.max(abs(actual - reference))))
                raw_maximum = max(raw_maximum, float(np.max(abs(rotation_increment(q, delta, point) - reference))))
                if scale == 0:
                    np.testing.assert_array_equal(actual, np.zeros(3, dtype=np.float32))
            reports.append({"scale": scale, "mode": mode, "max_error_m": maximum, "raw_increment_error_m": raw_maximum})
    Path("/tmp/colibri_normalized_rotation_increment.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
