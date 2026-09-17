# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise Colibri gear contacts at the authored collision update rate."""

import unittest
from pathlib import Path

import warp as wp

import newton.examples
from newton.examples.kamino.example_kamino_colibri import Example as AssemblyChecks
from newton.examples.phoenx.example_phoenx_colibri import Example
from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Phoenx Colibri requires CUDA")
class TestColibriContacts(unittest.TestCase):
    def _make_example(self):
        assets = Path(newton.examples.get_asset_directory()) / "colibri"
        if not assets.is_dir():
            self.skipTest("Colibri mesh assets are not installed")
        args = Example.create_parser().parse_args(
            ["--contact-updates-per-frame", "2", "--substeps", "24", "--counterweight-density-scale", "1.0"]
        )
        return Example(ViewerNull(), args)

    def test_gear_pins_at_120_hz(self):
        """Keep the gear engaged during the initial passive-frame motion."""
        example = self._make_example()
        # Sticky normal geometry previously crossed 1 mm after 0.47 seconds.
        # Check the whole trajectory: final depths can hide a skipped pin.
        for _ in range(60):
            example.step()
            example.test_post_step()

    def test_gear_pins_for_one_minute_at_120_hz(self):
        """Reject intermittent pin crossings without accepting a stalled mechanism."""
        example = self._make_example()
        for _ in range(3600):
            example.step()
            AssemblyChecks.test_post_step(example)
            example._test_drive_tracking()
            depth, labels, _gear_depth = example._measure_contact_penetration()
            self.assertLess(
                depth,
                0.001,
                f"Penetration at {example.sim_time:.6f} s between {labels}: {depth:.6f} m",
            )
        # Support stationarity remains a separate, currently failing acceptance
        # check. Do not disable or relax it in the example to pass this test.


if __name__ == "__main__":
    unittest.main()
