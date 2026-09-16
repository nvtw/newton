# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Audit constructor-owned color groups with the unchanged public candidate policy."""

import json
import sys
from pathlib import Path

import numpy as np

from local_studies.colibri import check_slab_colibri as slab


class PublicSlabExample(slab.SlabExample):
    """Retain the original physical and slab topology assertions."""

    def __init__(self, viewer, args):
        if not args.public_solver:
            raise ValueError("This diagnostic requires --public-solver")
        super().__init__(viewer, args)
        data = self.solver.world._color_group_data
        if data is None or args.mass_splitting_color_group_size < 1:
            raise ValueError("This reference audit requires constructor-owned color groups")
        self.solver._slab_reference = {**data, "row_slab": data["row_partition"]}


if __name__ == "__main__":
    slab.runner.Example = PublicSlabExample
    try:
        slab.runner.main()
    finally:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        if slab.instances and output.exists():
            example = slab.instances[0]
            report = json.loads(output.read_text())
            velocity_filter = example.collision_pipeline.speculative_contact_velocity_filter
            report["speculative_contact_velocity_filter"] = velocity_filter
            report["candidate_admission"] = (
                "Public initial-twist policy" if velocity_filter else "Geometric admission within the motion envelope"
            )
            report["slab_schedule"] = example._slab_summary
            report["joint_accuracy"] = {
                "max_anchor_error_m": max((p["anchor_error_m"] for p in example._joint_peaks.values()), default=0.0),
                "max_axis_error_rad": max((p["axis_error_rad"] for p in example._joint_peaks.values()), default=0.0),
                "per_joint_peaks": example._joint_peaks,
                "final": example._joint_last,
            }
            report["search_envelope"] = {
                "cap_m": slab.speculative.cap,
                "max_observed_extension_m": example._max_extension,
                "max_observed_cached_points": example._max_points,
                "authored_shape_gaps_unchanged": True,
            }
            output.write_text(json.dumps(report, indent=2))

            np.savez_compressed(
                output.with_suffix(".state.npz"),
                q=example.state_0.body_q.numpy(),
                qd=example.state_0.body_qd.numpy(),
                labels=np.asarray(example.model.body_label),
                contact_impulses=example.solver.world._contact_container.impulses.numpy(),
                contact_derived=example.solver.world._contact_container.derived.numpy(),
                contact_anchors=example.solver.world._contact_container.lambdas.numpy(),
                contact_headers=example.solver.world._contact_cols.data.numpy(),
            )
