# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Audit actual source-contact effective masses against direct FP64 mobility."""

import json

import numpy as np

reference = np.load("/tmp/colibri_splitprep_trace_reference_inertia.npz")
candidate = np.load("/tmp/colibri_splitprep_trace_candidate.npz")
# Both processes enter preparation from exactly the same full source state.
# Compare raw bytes because column headers intentionally pack integer sentinels
# into float rows, which may appear as NaNs under floating-point comparison.
for key in candidate.files:
    if key.startswith("0_before_"):
        np.testing.assert_array_equal(reference[key].view(np.uint8), candidate[key].view(np.uint8))
    if key.startswith("0_after_") and not key.endswith("derived"):
        np.testing.assert_array_equal(reference[key].view(np.uint8), candidate[key].view(np.uint8))
np.testing.assert_array_equal(reference["0_after_derived"][3:], candidate["0_after_derived"][3:])
mass_before = reference["0_after_derived"][:3].view(np.int32).astype(np.int64)
mass_after = candidate["0_after_derived"][:3].view(np.int32).astype(np.int64)
assert np.max(np.abs(mass_before - mass_after)) <= 3

prefix = "0_after_"
columns = reference[prefix + "columns"].copy().view(np.int32)
count = int(reference[prefix + "column_count"][0])
geometry = reference[prefix + "derived"]
directions = reference[prefix + "anchors"]
inverse_mass = reference[prefix + "inverse_mass"].astype(np.float64)
sym = reference[prefix + "inverse_inertia_world"].astype(np.float64)
inertia = sym[:, [0, 3, 4, 3, 1, 5, 4, 5, 2]].reshape(-1, 3, 3)
exact = []
observed = [[], []]
for cid in range(count):
    b0, b1 = columns[1:3, cid]
    first, points = columns[5:7, cid]
    scale0, scale1 = columns[9:11, cid]
    for point in range(first, first + points):
        n = directions[:3, point].astype(np.float64)
        t = directions[3:6, point].astype(np.float64)
        r0 = geometry[9:12, point].astype(np.float64)
        r1 = geometry[12:15, point].astype(np.float64)
        for row, axis in enumerate((n, t, np.cross(n, t))):
            a = np.cross(r0, axis)
            b = np.cross(r1, axis)
            mobility = (
                scale0 * inverse_mass[b0]
                + scale1 * inverse_mass[b1]
                + scale0 * (a @ inertia[b0] @ a)
                + scale1 * (b @ inertia[b1] @ b)
            )
            exact.append(1.0 / mobility if mobility > 1.0e-12 else 0.0)
            observed[0].append(geometry[row, point])
            observed[1].append(candidate[prefix + "derived"][row, point])
exact = np.asarray(exact)
report = {"rows": len(exact)}
for label, values in zip(("reference", "candidate"), observed, strict=True):
    actual = np.asarray(values)
    relative = np.abs(actual.astype(np.float64) - exact) / np.maximum(np.abs(exact), 1.0e-300)
    report[label] = {
        "max_relative_error": float(relative.max()),
        "mean_relative_error": float(relative.mean()),
        "max_absolute_error": float(np.max(np.abs(actual - exact))),
    }
    # This is the FP32 forward-error bound for the scalar cross/matvec/dot
    # evaluation, independent of trajectory acceptance or physical tolerances.
    assert np.all(relative < 16 * np.finfo(np.float32).eps)
print(json.dumps(report, indent=2))
