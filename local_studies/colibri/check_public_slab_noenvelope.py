# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Use public solver construction with a local slab topology and the unchanged public candidate policy."""

import json
import sys
from pathlib import Path

import numpy as np

from local_studies.colibri import check_slab_colibri as slab
from local_studies.colibri.slab_parallel_prepare import install as install_parallel


class PublicSlabExample(slab.SlabExample):
    """Retain the original physical and slab topology assertions."""

    def __init__(self, viewer, args):
        if not args.public_solver:
            raise ValueError("This diagnostic requires --public-solver")
        super().__init__(viewer, args)
        self.solver._slab_reference = slab.install(self.solver, colors_per_slab=8, gpu=True)
        if args.parallel_prepare:
            install_parallel(self.solver)


if __name__ == "__main__":
    slab.runner.Example = PublicSlabExample
    try:
        slab.runner.main()
    finally:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        if slab.instances and output.exists():
            example = slab.instances[0]
            report = json.loads(output.read_text())
            report["candidate_admission"] = "Unchanged public initial-twist policy"
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
