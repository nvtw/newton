# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Captured source-scene slab screen with continuous topology verification."""

import json
import sys
from pathlib import Path

import numpy as np

from local_studies.colibri import check_chunk_coloring  # noqa: F401
from local_studies.colibri import check_speculative_tail as speculative
from local_studies.colibri.joint_accuracy import measure
from local_studies.colibri.slab_adapter import install
from local_studies.colibri.slab_schedule import build_schedule

runner = speculative.runner
original_install = runner.install_fused
instances = []


def install_fused(solver, *args, **kwargs):
    original_install(solver, *args, **kwargs)
    solver._slab_reference = install(solver, colors_per_slab=8, gpu=True)


class SlabExample(speculative.SpeculativeExample):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self._slab_summary = None
        self._joint_peaks = {}
        self._joint_last = None
        self._slab_audit_started = False
        instances.append(self)

    def test_post_step(self):
        super().test_post_step()
        self._joint_last = measure(self.state_0.body_q.numpy(), self.model.body_label)
        for row in self._joint_last["joints"]:
            peaks = self._joint_peaks.setdefault(row["joint"], {"anchor_error_m": 0.0, "axis_error_rad": 0.0})
            for key in peaks:
                peaks[key] = max(peaks[key], row.get(key, 0.0))
        # The authored-state check runs before the first captured schedule.
        if not self._slab_audit_started:
            self._slab_audit_started = True
            return
        world = self.solver.world
        data = self.solver._slab_reference
        n = int(world._num_active_constraints.numpy()[0])
        expected = build_schedule(
            world._elements.numpy()["bodies"][:n], world.num_bodies, world.mass_splitting_color_group_size or 8
        )
        np.testing.assert_array_equal(data["row_color"].numpy()[:n], expected.row_color)
        np.testing.assert_array_equal(data["row_slab"].numpy()[:n], expected.row_slab)
        np.testing.assert_array_equal(data["ids"].numpy()[:n], [r for c in expected.colors for r in c])
        np.testing.assert_array_equal(world._copy_state.count_per_node.numpy(), [len(s) for s in expected.body_slabs])
        self._slab_summary = expected.summary()


if __name__ == "__main__":
    runner.install_fused = install_fused
    runner.Example = SlabExample
    try:
        runner.main()
    finally:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        if instances and output.exists():
            example = instances[0]
            report = json.loads(output.read_text())
            report["slab_schedule"] = example._slab_summary
            report["joint_accuracy"] = {
                "max_anchor_error_m": max((p["anchor_error_m"] for p in example._joint_peaks.values()), default=0.0),
                "max_axis_error_rad": max((p["axis_error_rad"] for p in example._joint_peaks.values()), default=0.0),
                "per_joint_peaks": example._joint_peaks,
                "final": example._joint_last,
            }
            report["search_envelope"] = {
                "cap_m": speculative.cap,
                "max_observed_extension_m": example._max_extension,
                "max_observed_cached_points": example._max_points,
                "authored_shape_gaps_unchanged": True,
            }
            output.write_text(json.dumps(report, indent=2))
