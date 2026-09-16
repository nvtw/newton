# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check that invalid frozen inputs cannot publish a QR as valid."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.benchmark_tree_prepare_gpu import prepare_and_factor


def main():
    wp.init()
    data = dict(np.load("/tmp/colibri_joint_tree_mobility.npz"))
    cases = {}
    for name in ("negative_mass", "zero_mass", "nan_mass", "infinite_mass", "zero_joint_row", "nan_joint_row"):
        mass = data["reduced_mass"].copy()
        rows = data["loop_velocity_rows"].copy()
        drive = data["drive_sqrt_rows"].copy()
        if name == "negative_mass":
            mass[0, 0] = -abs(mass[0, 0])
        elif name == "zero_mass":
            mass[0] = 0
            mass[:, 0] = 0
        elif name == "nan_mass":
            mass[0, 0] = np.nan
        elif name == "infinite_mass":
            mass[0, 0] = np.inf
        elif name == "zero_joint_row":
            rows[0] = 0
            drive[0] = 0
        elif name == "nan_joint_row":
            rows[0, 0] = np.nan
        source = [wp.array(x.ravel(), dtype=wp.float64, device="cuda:0") for x in (mass, rows, drive)]
        output = wp.empty(48 * 14, dtype=wp.float64, device="cuda:0")
        status = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        chol = wp.empty(46 * 46, dtype=wp.float64, device="cuda:0")
        q = wp.full(48 * 48, 17.0, dtype=wp.float64, device="cuda:0")
        r = wp.full(48 * 14, 19.0, dtype=wp.float64, device="cuda:0")
        wp.launch(
            prepare_and_factor, dim=256, block_dim=256, inputs=[*source, output, status, chol, q, r], device="cuda:0"
        )
        cases[name] = {
            "rejected": bool(status.numpy()[0] == 0),
            "qr_untouched": bool(np.all(q.numpy() == 17) and np.all(r.numpy() == 19)),
        }
    Path("/tmp/colibri_tree_prepare_rejections.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(json.dumps(cases, indent=2))
    assert all(value["rejected"] and value["qr_untouched"] for value in cases.values())


if __name__ == "__main__":
    main()
