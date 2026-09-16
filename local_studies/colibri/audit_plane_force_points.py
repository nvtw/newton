# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Inspect saved common force-point heights; changed operators are diagnostics only."""

import json
from pathlib import Path

import numpy as np

from .check_friction_differential_reference import normalize, rotate


def interval(values):
    """Return min/max/span for geometric measurements."""
    return [float(np.min(values)), float(np.max(values)), float(np.ptp(values))]


def tangent_rows(levers, axes):
    """Build independent unsquared point tangent Jacobians."""
    return np.concatenate(
        [np.c_[basis[1:], np.cross(lever, basis[1:])] for lever, basis in zip(levers, axes, strict=True)]
    )


def main():
    """Compare native levers, rebuilt midpoint, and explicit plane-point conventions."""
    path = "/tmp/colibri_analytical_birth_capture.birth.npz"
    d = np.load(path)
    count = int(d["rigid_contact_count"][0])
    ids = d["shape_body"][d["rigid_contact_shape0"][:count]] + 1
    ground = d["shape_body"][d["rigid_contact_shape1"][:count]] + 1
    assert np.all(ground == 0)
    plane_height = float(np.float32(0.002965275))
    normal = d["lambdas"][:3, :count].T.astype(float)
    tangent = d["lambdas"][3:6, :count].T.astype(float)
    axes = np.stack([normal, tangent, np.cross(normal, tangent)], axis=1)
    w0, w1 = [], []
    for k, body in enumerate(ids):
        q = normalize(d["orientation"][body].astype(float))
        v = d["rigid_contact_point0"][k].astype(float) - d["body_com"][body].astype(float)
        w0.append(d["position"][body].astype(float) + rotate(q, v) + float(d["rigid_contact_margin0"][k]) * normal[k])
        w1.append(d["rigid_contact_point1"][k].astype(float) - float(d["rigid_contact_margin1"][k]) * normal[k])
    w0, w1 = np.array(w0), np.array(w1)
    midpoint = 0.5 * (w0 + w1)
    native_r0 = d["derived"][9:12, :count].T.astype(float)
    native_r1 = d["derived"][12:15, :count].T.astype(float)
    com = d["position"][ids].astype(float)
    native_point = com + native_r0
    plane = midpoint.copy()
    plane[:, 2] = plane_height
    # Explicit FP32 construction, not a high-precision production proposal.
    plane32 = plane.astype(np.float32)
    r_plane32 = (plane32 - d["position"][ids]).astype(float)
    r_ground32 = (plane32 - d["position"][0]).astype(float)
    reports = []
    for body in np.unique(ids):
        mask = ids == body
        basis = axes[mask]
        native = tangent_rows(native_r0[mask], basis)
        mid = tangent_rows(midpoint[mask] - com[mask], basis)
        projected = tangent_rows(r_plane32[mask], basis)
        n_before = np.cross(native_r0[mask], normal[mask])
        n_after = np.cross(r_plane32[mask], normal[mask])
        # Isolate height-only effect, preserving original x/y lever entries exactly.
        rz_only = native_r0[mask].copy()
        rz_only[:, 2] = np.float32(plane_height) - d["position"][body, 2]
        projected_height_only = tangent_rows(rz_only, basis)
        defect_before = np.cross(native_r1[mask] - native_point[mask], basis[:, 1])
        defect_after = np.cross(r_ground32[mask] - (com[mask] + r_plane32[mask]), basis[:, 1])
        reports.append(
            {
                "body": int(body),
                "contacts": int(mask.sum()),
                "body_surface_height_m": interval(w0[mask, 2]),
                "ground_witness_height_m": interval(w1[mask, 2]),
                "native_common_height_m": interval(native_point[mask, 2]),
                "rebuilt_midpoint_height_m": interval(midpoint[mask, 2]),
                "native_surface_gap_m": interval(w0[mask, 2] - w1[mask, 2]),
                "native_two_side_common_point_error_m": float(np.max(abs(native_r1[mask] - native_point[mask]))),
                "native_vs_rebuilt_midpoint_error_m": float(np.max(abs(native_point[mask] - midpoint[mask]))),
                "normal_angular_row_change_all_rebuilt": float(np.max(abs(n_after - n_before))),
                "normal_angular_row_change_height_only": float(np.max(abs(np.cross(rz_only, normal[mask]) - n_before))),
                "tangent_singular_native": np.linalg.svd(native, compute_uv=False).tolist(),
                "tangent_singular_rebuilt_midpoint": np.linalg.svd(mid, compute_uv=False).tolist(),
                "tangent_singular_plane_fp32": np.linalg.svd(projected, compute_uv=False).tolist(),
                "tangent_singular_height_only_fp32": np.linalg.svd(projected_height_only, compute_uv=False).tolist(),
                "tangent_operator_max_change_height_only": float(np.max(abs(projected_height_only - native))),
                "torque_defect_per_unit_tangent_before": float(np.max(abs(defect_before))),
                "torque_defect_per_unit_tangent_plane_fp32": float(np.max(abs(defect_after))),
            }
        )
    report = {
        "capture": path,
        "ground_height_float32_m": plane_height,
        "ground_source": "example_kamino_colibri.py:2346, add_ground_plane(height=0.002965275)",
        "scope": "Plane projection is an explicitly changed force-point operator, not a rank truncation or native result.",
        "bodies": reports,
    }
    Path("/tmp/colibri_plane_force_points.json").write_text(json.dumps(report, indent=2))
    np.savez(
        "/tmp/colibri_plane_force_points.npz",
        body_ids=ids,
        axes=axes,
        native_r0=native_r0,
        native_r1=native_r1,
        surface0=w0,
        surface1=w1,
        midpoint=midpoint,
        plane=plane32,
        plane_r0=r_plane32,
        plane_r1=r_ground32,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
