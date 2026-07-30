# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Configuration regressions for PhoenX's inequality-only PGS path."""

from __future__ import annotations

import unittest

import warp as wp

from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene

SUBSTEPS = 5
SETTLE_FRAMES = 60


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX PGS tests require CUDA graphs.")
class TestColoringPriorityBias(unittest.TestCase):
    """Check contact coloring without any joint columns."""

    def test_contacts_appear_in_first_colour_when_no_joints(self) -> None:
        """Place a single contact column in the first color."""
        scene = _PhoenXScene(substeps=SUBSTEPS)
        scene.add_ground_plane()
        scene.add_box(position=(0.0, 0.0, 0.5), half_extents=(0.5, 0.5, 0.5))
        scene.finalize()
        for _ in range(SETTLE_FRAMES):
            scene.step()
        self.assertGreaterEqual(int(scene.world._world_num_colors.numpy()[0]), 1)
        starts = scene.world._world_color_starts.numpy()
        element_ids = scene.world._world_element_ids_by_color.numpy()
        start, end = int(starts[0, 0]), int(starts[0, 1])
        self.assertGreater(end - start, 0)
        for index in range(start, end):
            self.assertGreaterEqual(int(element_ids[index]), 0)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX PGS tests require CUDA graphs.")
class TestVelocityIterationsValidator(unittest.TestCase):
    """Validate optional contact relaxation iteration counts."""

    def test_negative_velocity_iters_rejected(self) -> None:
        """Reject a negative number of velocity iterations."""
        with self.assertRaises(ValueError) as context:
            _PhoenXScene(
                substeps=SUBSTEPS,
                solver_iterations=8,
                velocity_iterations=-1,
            ).finalize()
        self.assertIn("velocity_iterations", str(context.exception))

    def test_zero_velocity_iters_settles_box(self) -> None:
        """Settle a resting contact without the optional relaxation pass."""
        scene = _PhoenXScene(
            substeps=SUBSTEPS,
            solver_iterations=8,
            velocity_iterations=0,
        )
        scene.add_ground_plane()
        scene.add_box(position=(0.0, 0.0, 0.5), half_extents=(0.5, 0.5, 0.5))
        scene.finalize()
        for _ in range(SETTLE_FRAMES):
            scene.step()
        self.assertLess(abs(float(scene.body_position(0)[2]) - 0.5), 0.05)


if __name__ == "__main__":
    unittest.main()
