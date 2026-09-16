"""FP32 differential material reference versus an offline FP64 geometry oracle."""

import json
from pathlib import Path
import numpy as np


def normalize(q):
    return q / np.sqrt(q @ q)


def conjugate(q):
    return q * np.array([-1, -1, -1, 1], dtype=q.dtype)


def multiply(a, b):
    return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - a[:3] @ b[:3]].astype(a.dtype)


def rotate(q, v):
    t = np.cross(q[:3], v)
    return v + q.dtype.type(2) * (q[3] * t + np.cross(q[:3], t))


def rotation_difference(qbirth, deltaq, v):
    old_cross = np.cross(qbirth[:3], v)
    delta_cross = np.cross(deltaq[:3], v)
    current = qbirth + deltaq
    return qbirth.dtype.type(2) * (
        deltaq[3] * old_cross
        + current[3] * delta_cross
        + np.cross(deltaq[:3], old_cross)
        + np.cross(current[:3], delta_cross)
    )


def differential(pbirth, q0birth, q1birth, pnow, q0now, q1now, anchor):
    q0birth, q1birth, q0now, q1now = map(normalize, (q0birth, q1birth, q0now, q1now))
    dq0 = q0now - q0birth
    dq1 = q1now - q1birth
    relbirth = multiply(conjugate(q0birth), q1birth)
    delta_rel = multiply(conjugate(dq0), q1birth) + multiply(conjugate(q0now), dq1)
    delta_t = rotate(conjugate(q0now), pnow - pbirth) + rotation_difference(conjugate(q0birth), conjugate(dq0), pbirth)
    return rotate(q0now, delta_t + rotation_difference(relbirth, delta_rel, anchor))


def direct(pbirth, q0birth, q1birth, pnow, q0now, q1now, anchor):
    q0birth, q1birth, q0now, q1now = map(normalize, (q0birth, q1birth, q0now, q1now))
    t = rotate(conjugate(q0now), pnow) - rotate(conjugate(q0birth), pbirth)
    dr = rotate(multiply(conjugate(q0now), q1now), anchor) - rotate(multiply(conjugate(q0birth), q1birth), anchor)
    return rotate(q0now, t + dr)


def main():
    rng = np.random.default_rng(69211)
    errors = []
    naive = []
    covariance = []
    birth = []
    for _ in range(500):
        q0 = normalize(rng.normal(size=4)).astype(np.float32)
        q1 = normalize(rng.normal(size=4)).astype(np.float32)
        p = rng.normal(size=3).astype(np.float32) * np.float32(0.2)
        a = rng.normal(size=3).astype(np.float32) * np.float32(0.1)
        scale = 10 ** rng.uniform(-7, -2)
        q0n = normalize(q0 + rng.normal(size=4).astype(np.float32) * np.float32(scale))
        q1n = normalize(q1 + rng.normal(size=4).astype(np.float32) * np.float32(scale))
        pn = p + rng.normal(size=3).astype(np.float32) * np.float32(scale)
        args = (p, q0, q1, pn, q0n, q1n, a)
        out = differential(*args)
        ref = direct(*(x.astype(float) for x in args))
        errors.append(float(np.max(abs(out - ref))))
        naive.append(float(np.max(abs(direct(*args) - ref))))
        birth.append(float(np.max(abs(differential(p, q0, q1, p, q0, q1, a)))))
        g = normalize(rng.normal(size=4)).astype(np.float32)
        transformed = (
            rotate(g, p),
            multiply(g, q0),
            multiply(g, q1),
            rotate(g, pn),
            multiply(g, q0n),
            multiply(g, q1n),
            a,
        )
        covariance.append(float(np.max(abs(differential(*transformed) - rotate(g, out)))))
    report = {
        "cases": len(errors),
        "all_arithmetic": "FP32 candidate; FP64 only independent offline reference",
        "birth_max_m": max(birth),
        "differential_max_error_m": max(errors),
        "naive_max_error_m": max(naive),
        "differential_median_error_m": float(np.median(errors)),
        "naive_median_error_m": float(np.median(naive)),
        "global_rotation_covariance_max_m": max(covariance),
        "storage": "11 FP32 reference values: relative COM translation plus both birth quaternions; existing one local anchor",
        "scope": "Numerical formulation only; no contact matching, native solve, or live trajectory yet",
    }
    assert max(birth) == 0
    Path("/tmp/colibri_friction_differential_reference.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
