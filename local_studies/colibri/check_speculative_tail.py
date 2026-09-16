# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Opt in to existing velocity-derived search gaps on the exact-offset scene."""

import json
import sys
from pathlib import Path

import numpy as np

from local_studies.colibri import check_bilateral_pgs as runner

cap = 0.005
if "--search-cap" in sys.argv:
    index = sys.argv.index("--search-cap")
    cap = float(sys.argv[index + 1])
    del sys.argv[index : index + 2]

original = runner.Example
instances = []


class SpeculativeExample(original):
    def __init__(self, viewer, args):
        args.speculative_contact_gap_max = cap
        super().__init__(viewer, args)
        self._base_gaps = self.model.shape_gap.numpy().copy()
        self._max_extension = 0.0
        self._max_points = 0
        instances.append(self)

    def test_post_step(self):
        pipeline = self.collision_pipeline
        actual = pipeline._shape_search_gap.numpy() - self._base_gaps
        self._max_extension = max(self._max_extension, float(np.max(actual)))
        self._max_points = max(self._max_points, int(self.contacts.rigid_contact_count.numpy()[0]))
        np.testing.assert_array_equal(self.model.shape_gap.numpy(), self._base_gaps)
        super().test_post_step()


if __name__ == "__main__":
    runner.Example = SpeculativeExample
    try:
        runner.main()
    finally:
        if instances:
            example = instances[0]
            output = Path(sys.argv[sys.argv.index("--output") + 1])
            report = json.loads(output.read_text())
            report["search_envelope"] = {
                "cap_m": cap,
                "max_observed_extension_m": example._max_extension,
                "max_observed_cached_points": example._max_points,
                "capacity": example.contact_capacity,
                "authored_shape_gaps_unchanged": True,
            }
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report["search_envelope"]))
