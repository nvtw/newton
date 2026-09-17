# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise Colibri gear contacts at the authored collision update rate."""

import unittest
from pathlib import Path

import warp as wp

import newton.examples
from newton.examples.phoenx.example_phoenx_colibri import Example
from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Phoenx Colibri requires CUDA")
class TestColibriContacts(unittest.TestCase):
    def test_gear_pins_at_120_hz(self):
        """Keep the gear engaged during the initial passive-frame motion."""
        assets = Path(newton.examples.get_asset_directory()) / "colibri"
        if not assets.is_dir():
            self.skipTest("Colibri mesh assets are not installed")
        args = Example.create_parser().parse_args(
            ["--contact-updates-per-frame", "2", "--substeps", "24", "--counterweight-density-scale", "1.0"]
        )
        example = Example(ViewerNull(), args)
        # Sticky normal geometry previously crossed 1 mm after 0.47 seconds.
        # Check the whole trajectory: final depths can hide a skipped pin.
        for _ in range(60):
            example.step()
            example.test_post_step()


if __name__ == "__main__":
    unittest.main()
