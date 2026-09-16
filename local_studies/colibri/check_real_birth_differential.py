# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Audit raw FP32 differential history; doubles occur only in offline oracles."""

import json
from pathlib import Path

import numpy as np

from .check_friction_differential_reference import conjugate, multiply, normalize, rotate
from .check_quaternion_increment_math import rotation_increment


def candidate(p0, p1, q0, q1, n0, n1, r0, r1, anchor):
    """Evaluate the fourteen-FP32-value differential reference without normalization."""
    dq0, dq1 = r0 - q0, r1 - q1
    qr = multiply(conjugate(q0), q1)
    dqr = multiply(conjugate(dq0), q1) + multiply(conjugate(r0), dq1)
    dt = rotate(conjugate(r0), (n1 - p1) - (n0 - p0))
    dt += rotation_increment(conjugate(q0), conjugate(dq0), p1 - p0)
    result = rotate(r0, dt + rotation_increment(qr, dqr, anchor))
    assert result.dtype == np.float32
    return result


def direct(args, unit=False):
    """Subtract absolute relative transforms, with optional oracle normalization."""
    p0, p1, q0, q1, n0, n1, r0, r1, anchor = args
    if unit:
        q0, q1, r0, r1 = map(normalize, (q0, q1, r0, r1))
    dt = rotate(conjugate(r0), n1 - n0) - rotate(conjugate(q0), p1 - p0)
    dr = rotate(multiply(conjugate(r0), r1), anchor) - rotate(multiply(conjugate(q0), q1), anchor)
    return rotate(r0, dt + dr)


def transform(args, q, offset):
    """Change the world frame before rounding stored states."""
    p0, p1, q0, q1, n0, n1, r0, r1, anchor = args
    return (
        rotate(q, p0) + offset,
        rotate(q, p1) + offset,
        multiply(q, q0),
        multiply(q, q1),
        rotate(q, n0) + offset,
        rotate(q, n1) + offset,
        multiply(q, r0),
        multiply(q, r1),
        anchor,
    )


def norm(x):
    """Return a scalar maximum absolute component."""
    return float(np.max(abs(x)))


def main():
    """Separate arithmetic cancellation from quantization of the actual pose state."""
    d = np.load("/tmp/colibri_analytical_birth_capture.birth.npz")
    rng = np.random.default_rng(41982)
    fixtures = []
    for i in range(int(d["rigid_contact_count"][0])):
        body = int(d["shape_body"][int(d["rigid_contact_shape0"][i])]) + 1
        if body <= 0:
            continue
        anchor = d["rigid_contact_point0"][i] - d["body_com"][body]
        for other in (0, 2 if body == 1 else 1):
            fixtures.append(
                tuple(
                    x.astype(float)
                    for x in (
                        d["position"][other],
                        d["position"][body],
                        d["orientation"][other],
                        d["orientation"][body],
                        anchor,
                    )
                )
            )
    rows, zero_max = [], 0.0
    for scale in (1e-9, 1e-7, 1e-5, 1e-3):
        for mode in ("translation", "rotation", "mixed", "common_rigid"):
            values = []
            for p0, p1, q0, q1, anchor in fixtures:
                t = rng.normal(size=(2, 3)) * scale
                dq = np.c_[rng.normal(size=(2, 3)) * (0.5 * scale), np.ones(2)]
                dq = np.array([normalize(v) for v in dq])
                n0, n1, r0, r1 = p0.copy(), p1.copy(), q0.copy(), q1.copy()
                if mode in ("translation", "mixed"):
                    n0, n1 = p0 + t[0], p1 + t[1]
                if mode in ("rotation", "mixed"):
                    r0, r1 = multiply(dq[0], q0), multiply(dq[1], q1)
                if mode == "common_rigid":
                    n0, n1 = rotate(dq[0], p0) + t[0], rotate(dq[0], p1) + t[0]
                    r0, r1 = multiply(dq[0], q0), multiply(dq[0], q1)
                ideal = (p0, p1, q0, q1, n0, n1, r0, r1, anchor)
                inputs = tuple(v.astype(np.float32) for v in ideal)
                stored = tuple(v.astype(float) for v in inputs)
                out, raw, unit = candidate(*inputs), direct(stored), direct(stored, True)
                truth = direct(ideal, True)
                zero = (*inputs[:4], *inputs[:4], inputs[-1])
                zero_max = max(zero_max, norm(candidate(*zero)))
                stats = {
                    "arithmetic": norm(out - raw),
                    "naive": norm(direct(inputs) - raw),
                    "polynomial_vs_unit": norm(raw - unit),
                    "state_quantization": norm(unit - truth),
                    "total_error": norm(out - truth),
                    "intended_displacement": norm(truth),
                }
                worldq = normalize(rng.normal(size=4))
                for shift in (0.0, 10.0):
                    transformed = transform(ideal, worldq, np.array([shift, -shift, shift]))
                    changed = tuple(v.astype(np.float32) for v in transformed)
                    changed64 = tuple(v.astype(float) for v in changed)
                    cout, craw, cunit = candidate(*changed), direct(changed64), direct(changed64, True)
                    key = "rotated" if shift == 0 else "translated_10m"
                    stats[key + "_arithmetic"] = norm(cout - craw)
                    stats[key + "_ideal_oracle_covariance"] = norm(direct(transformed, True) - rotate(worldq, truth))
                    stats[key + "_covariance"] = norm(cout - rotate(worldq, out.astype(float)))
                    stats[key + "_oracle_quantization_covariance"] = norm(cunit - rotate(worldq, unit))
                values.append(stats)
            rows.append(
                {
                    "scale": scale,
                    "mode": mode,
                    "cases": len(values),
                    "maximum_m": {k: max(v[k] for v in values) for k in values[0]},
                }
            )
    report = {
        "candidate_dtype": "FP32; doubles only offline oracle/control generation",
        "fixtures": len(fixtures),
        "birth_zero_m": zero_max,
        "cases": rows,
        "scope": "Real birth geometry, synthetic motion; no native lifecycle or live acceptance.",
    }
    assert zero_max == 0
    for row in rows:
        errors = row["maximum_m"]
        assert errors["rotated_ideal_oracle_covariance"] < 1e-13
        assert errors["translated_10m_ideal_oracle_covariance"] < 1e-13
        if row["scale"] <= 1e-7:
            assert errors["arithmetic"] < 1e-12
        if row["mode"] == "common_rigid":
            assert errors["intended_displacement"] < 1e-13
    Path("/tmp/colibri_real_birth_differential.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
