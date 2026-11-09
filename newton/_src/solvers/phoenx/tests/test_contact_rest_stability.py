# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resting-stack stability of PhoenX contacts under few solver sweeps.

A short Kapla wall of planks standing on their 8 mm edges, solved like the
Kapla arena (4 substeps, 4 iterations, 1 relax iteration). With the contact
spring at the Nyquist limit the unconverged sweeps feed penetration
corrections into the planks' rocking modes, and the resting wall jitters at
about 1 cm/s; a stable resting state stays below 1 mm/s.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene

#: Kapla plank half extents [m]: 8 mm thick, 100 mm long, 25 mm tall.
_PLANK = (0.004, 0.05, 0.0125)


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX contact tests require CUDA graph capture")
class TestContactRestStability(unittest.TestCase):
    def test_edge_standing_plank_wall_stays_at_rest(self) -> None:
        """Six layers of crossing on-edge planks must not start rocking."""
        scene = _PhoenXScene(
            fps=120,
            substeps=4,
            solver_iterations=4,
            velocity_iterations=1,
            friction=0.5,
            step_layout="single_world",
        )
        scene.add_ground_plane()
        along_x = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        ids = []
        for layer in range(6):
            z = _PLANK[2] * (2 * layer + 1)
            if layer % 2 == 0:
                # Three rows of two planks along x.
                for y in (-0.046, 0.0, 0.046):
                    for x in (-0.05, 0.05):
                        ids.append(scene.add_box((x, y, z), _PLANK, orientation=along_x))
            else:
                # Four planks across the rows.
                for x in (-0.075, -0.025, 0.025, 0.075):
                    ids.append(scene.add_box((x, 0.0, z), _PLANK))
        scene.finalize()
        initial = np.array([scene.body_position(i) for i in ids])

        speeds = []
        for frame in range(240):
            scene.step()
            if frame >= 60 and frame % 20 == 0:
                velocity = np.array([scene.body_velocity(i) for i in ids])
                speeds.append(np.linalg.norm(velocity, axis=1))
        speeds = np.array(speeds)
        final = np.array([scene.body_position(i) for i in ids])

        self.assertLess(float(np.max(np.median(speeds, axis=1))), 1.0e-3)
        self.assertLess(float(np.max(speeds)), 5.0e-3)
        self.assertLess(float(np.max(np.linalg.norm(final - initial, axis=1))), 1.0e-3)


if __name__ == "__main__":
    unittest.main()
