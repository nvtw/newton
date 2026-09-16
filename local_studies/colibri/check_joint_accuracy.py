# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Track source joint errors throughout the validated Colibri trajectory."""

import json
import runpy
import sys
from pathlib import Path

from local_studies.colibri import check_bilateral_pgs as runner
from local_studies.colibri.joint_accuracy import measure

_original = runner.Example
_instances = []


class _AccuracyExample(_original):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self._joint_peaks = {}
        self._joint_last = None
        self._joint_samples = 0
        _instances.append(self)

    def test_post_step(self):
        super().test_post_step()
        result = measure(self.state_0.body_q.numpy(), self.model.body_label)
        self._joint_last = result
        self._joint_samples += 1
        for row in result["joints"]:
            peaks = self._joint_peaks.setdefault(row["joint"], {"anchor_error_m": 0.0, "axis_error_rad": 0.0})
            peaks["anchor_error_m"] = max(peaks["anchor_error_m"], row["anchor_error_m"])
            peaks["axis_error_rad"] = max(peaks["axis_error_rad"], row.get("axis_error_rad", 0.0))


if __name__ == "__main__":
    if "--output" not in sys.argv:
        raise ValueError("Joint accuracy tracking requires an explicit --output path")
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    if "--split-parallel-prepare" in sys.argv:
        from local_studies.colibri.parallel_prepare_split import install

        sys.argv.remove("--split-parallel-prepare")
        runner.install_parallel_prepare = install
        if "--parallel-prepare" not in sys.argv:
            sys.argv.append("--parallel-prepare")
    runner.Example = _AccuracyExample
    try:
        runpy.run_module("local_studies.colibri.check_batch_aware_tail", run_name="__main__")
    finally:
        if _instances and output.exists():
            example = _instances[0]
            report = json.loads(output.read_text())
            report["joint_accuracy"] = {
                "samples": example._joint_samples,
                "max_anchor_error_m": max((p["anchor_error_m"] for p in example._joint_peaks.values()), default=0.0),
                "max_axis_error_rad": max((p["axis_error_rad"] for p in example._joint_peaks.values()), default=0.0),
                "per_joint_peaks": example._joint_peaks,
                "final": example._joint_last,
                "scope": "Source joint residuals at every frame including initial state; measurement excluded from physics timing",
            }
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in report["joint_accuracy"].items() if k not in ("per_joint_peaks", "final")})
            )
