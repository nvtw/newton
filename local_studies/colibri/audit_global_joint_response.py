# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Audit the high-precision frozen joint correction against physical budgets."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    """Measure momentum, work and held-contact residuals without live injection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction", type=Path, default=Path("/tmp/colibri_joint_original_corrected_response.npz"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/colibri_global_joint_physics_audit.json"))
    parser.add_argument("--operator", type=Path, default=Path("/tmp/colibri_global_joint_probe.npz"))
    args = parser.parse_args()
    s = dict(np.load("/tmp/colibri_support_relax330.npz"))
    p = dict(np.load(args.operator))
    correction = dict(np.load(args.correction))
    active = correction["active"]
    W = p["W"]
    M = np.zeros_like(W)
    M[active] = np.linalg.inv(W[active])
    v = p["initial"]
    dv = np.zeros_like(v)
    dv[active] = correction["dv"]
    # Independently enforce the correction sign using the physical equation,
    # not the candidate solver's own residual report or normal-matrix product.
    residual = p["rhs"] + np.einsum("rbi,bi->r", p["J"], dv) + p["compliance"] * correction["delta"]
    equation_error = float(np.max(np.abs(residual)))
    if equation_error > 1e-8:
        raise ValueError(f"Correction fails original constitutive equation: {equation_error:.9g}")
    after = v + dv
    physical_impulse = np.einsum("bij,bj->bi", M, dv)
    linear = np.sum(physical_impulse[:, :3], axis=0)
    angular = np.sum(physical_impulse[:, 3:] + np.cross(s["position"], physical_impulse[:, :3]), axis=0)
    # Report prescribed endpoint reaction separately, even though this capture's
    # joint graph contains no world joint and the flower is disconnected.
    row_impulse = np.einsum("rbi,r->bi", p["J"], correction["delta"])
    prescribed = np.flatnonzero(W[:, 0, 0] == 0)
    reaction_linear = row_impulse[prescribed, :3].sum(axis=0)
    reaction_angular = (
        row_impulse[prescribed, 3:] + np.cross(s["position"][prescribed], row_impulse[prescribed, :3])
    ).sum(axis=0)
    energy_change = float(np.einsum("bi,bij,bj->", after, M, after) / 2 - np.einsum("bi,bij,bj->", v, M, v) / 2)
    midpoint_work = float(np.sum(physical_impulse * (v + after) / 2))
    h = s["headers"].view(np.int32)

    def residuals(velocity):
        """Evaluate eligible native relaxation contacts with impulses held fixed."""
        normal, tangent = [], []
        for col in range(int(s["column_count"][0])):
            a, b = h[1:3, col]
            mu = float(s["headers"][4, col])
            for point in range(h[5, col], h[5, col] + h[6, col]):
                if s["derived"][3, point] > 0:
                    continue
                n, t = s["lambdas"][:3, point].astype(float), s["lambdas"][3:6, point].astype(float)
                axes = np.array([n, t, np.cross(n, t)])
                ja = -np.concatenate((axes, np.cross(s["derived"][9:12, point], axes)), axis=1)
                jb = np.concatenate((axes, np.cross(s["derived"][12:15, point], axes)), axis=1)
                K = ja @ W[a] @ ja.T + jb @ W[b] @ jb.T
                scale = float(np.max(np.diag(K)))
                g = ja @ velocity[a] + jb @ velocity[b]
                impulse = s["after_impulses"][:, point].astype(float)
                normal.append(max(-g[0], -impulse[0] * scale, abs(min(impulse[0] * scale, g[0])), 0))
                trial = impulse[1:] - g[1:] / scale
                radius = mu * max(impulse[0], 0)
                projection = trial * min(1, radius / max(np.linalg.norm(trial), 1e-300))
                tangent.append(float(np.linalg.norm(impulse[1:] - projection) * scale))
        return {
            "eligible_points": len(normal),
            "max_normal_residual_m_s": float(max(normal)),
            "max_tangent_residual_m_s": float(max(tangent)),
        }

    result = {
        "scope": "Frozen joint-only correction, held contact impulses, no live simulation",
        "correction": str(args.correction),
        "operator": str(args.operator),
        "original_constitutive_equation_max_residual": equation_error,
        "linear_momentum_change_plus_reaction_N_s": (linear + reaction_linear).tolist(),
        "angular_momentum_change_plus_reaction_N_m_s": (angular + reaction_angular).tolist(),
        "kinetic_energy_change_J": energy_change,
        "midpoint_impulse_work_J": midpoint_work,
        "work_identity_error_J": abs(energy_change - midpoint_work),
        "before_contacts": residuals(v),
        "after_contacts": residuals(after),
        "caveat": "Work identity alone is not a physical energy acceptance check. Momentum and contact criteria remain independent.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
