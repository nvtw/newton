"""All-FP32 six-coordinate solve proposal, audited against the saved FP64 physical operator."""

import json
from pathlib import Path

import numpy as np


def solve32(matrix, rhs):
    """Diagonal equilibration, positive Cholesky; no physical regularization."""
    scale = np.float32(1) / np.sqrt(np.diag(matrix))
    a = (scale[:, None] * matrix) * scale[None, :]
    b = scale * rhs
    l = np.zeros_like(a)
    for i in range(6):
        for k in range(i + 1):
            value = np.float32(a[i, k])
            for j in range(k):
                value = np.float32(value - np.float32(l[i, j] * l[k, j]))
            if i == k:
                if not value > 0:
                    raise ArithmeticError("UNKNOWN nonpositive Cholesky pivot")
                l[i, k] = np.sqrt(value)
            else:
                l[i, k] = np.float32(value / l[k, k])
    y = np.zeros(6, dtype=np.float32)
    for i in range(6):
        value = np.float32(b[i])
        for j in range(i):
            value = np.float32(value - np.float32(l[i, j] * y[j]))
        y[i] = np.float32(value / l[i, i])
    z = np.zeros(6, dtype=np.float32)
    for i in range(5, -1, -1):
        value = np.float32(y[i])
        for j in range(i + 1, 6):
            value = np.float32(value - np.float32(l[j, i] * z[j]))
        z[i] = np.float32(value / l[i, i])
    return scale * z


def accumulate32(terms):
    value = np.zeros_like(terms[0], dtype=np.float32)
    error = np.zeros_like(value)
    for term in terms:
        adjusted = np.asarray(term, dtype=np.float32) - error
        new = np.asarray(value + adjusted, dtype=np.float32)
        error = (new - value) - adjusted
        value = new
    return value


def proposal(x):
    ids = x["eligible"]
    a = x["point_map"].astype(np.float32)
    old = x["impulses"][ids].astype(np.float32).ravel()
    weights = np.repeat(x["impulses"][ids, 0].astype(np.float32), 3)
    weights = weights / np.max(weights)
    old_force = accumulate32([a[:, k] * old[k] for k in range(len(old))])
    requested = old_force - solve32(x["mobility"].astype(np.float32), x["velocity"][:6].astype(np.float32))
    gram = accumulate32([weights[k] * np.outer(a[:, k], a[:, k]) for k in range(len(old))])
    dual = solve32(gram, requested)
    proposed = np.array([weights[k] * accumulate32(a[:, k] * dual) for k in range(len(old))], dtype=np.float32)
    assert proposed.dtype == np.float32
    return proposed.reshape(-1, 3), {
        "equilibrated_gram_condition": float(np.linalg.cond(gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))))
    }


def audit(x, proposed):
    ids = x["eligible"]
    lam = x["impulses"].copy()
    lam[ids] = proposed
    c = x["contact_rows"]
    w = x["operator"]
    j = x["joints"]
    comp = x["compliance"]
    v = x["velocity"]
    force = c.reshape(-1, 12).T @ (lam - x["impulses"]).ravel()
    reaction = -np.linalg.solve(j @ w @ j.T + np.diag(comp), j @ w @ force)
    after = v + w @ (force + j.T @ reaction)
    cv = np.einsum("nki,i->nk", c, after)
    spec = x["derived"][3] > 0
    normal = float(np.max(np.where(lam[ids, 0] > 0, abs(cv[ids, 0]), np.maximum(-cv[ids, 0], 0))))
    tangent = float(np.max(abs(cv[ids, 1:])))
    cone = float(np.max(np.linalg.norm(lam[ids, 1:], axis=1) - x["coefficients"][ids] * lam[ids, 0]))
    drive = j @ after + comp * (x["drive_impulses"] + reaction) - x["drive_reference"]
    hard = float(np.max(abs((j @ after)[~x["dynamic"]])))
    hard_delta = float(np.max(abs((j @ (after - v))[~x["dynamic"]])))
    drive_error = float(np.max(abs(drive[x["dynamic"]])))
    spec_margin = float(np.min(cv[spec, 0] + x["derived"][3, spec]))
    contact_ok = bool(
        normal <= 1e-8
        and tangent <= 1e-8
        and cone <= 1e-12
        and min(lam[ids, 0]) >= 0
        and drive_error <= 1e-8
        and spec_margin >= -1e-8
        and hard_delta <= 1e-8
    )
    return {
        "accepted": bool(contact_ok and hard <= 1e-8),
        "contact_and_homogeneous_joint_certificate": contact_ok,
        "absolute_hard_joint_residual": hard,
        "hard_joint_delta_residual": hard_delta,
        "normal_residual_m_s": normal,
        "tangent_residual_m_s": tangent,
        "max_cone_excess_Ns": cone,
        "minimum_normal_Ns": float(min(lam[ids, 0])),
        "base_velocity": after[:6].tolist(),
        "drive_residual_rad_s": float(np.max(abs(drive[x["dynamic"]]))),
        "minimum_speculative_margin_m_s": float(np.min(cv[spec, 0] + x["derived"][3, spec])),
    }


def main():
    x = np.load("/tmp/colibri_coupled_normal_stick_reference.npz")
    proposed, info = proposal(x)
    report = audit(x, proposed)
    report.update(info)
    report["scope"] = __doc__
    Path("/tmp/colibri_full_wrench_fp32.json").write_text(json.dumps(report, indent=2))
    np.save("/tmp/colibri_full_wrench_fp32_impulses.npy", proposed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
