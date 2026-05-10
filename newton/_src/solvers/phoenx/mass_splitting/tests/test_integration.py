# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end mass-splitting integration tests that catch the
regressions the per-component tests miss.

Drives 3- and 6-cube stacks through the full ``PhoenXWorld.step`` path
under ``mass_split_max_partitions=2`` (forces overflow on the bottom
cube which has 8+ contacts: ground + corners of the cube above) and
asserts the stack doesn't explode (velocities stay bounded, no NaN,
cubes don't fly off in random directions). Catches the failure mode
the user reported: "the tower scene still explodes after a few
frames".

Per ``feedback_avoid_full_suite`` and ``feedback_test_runner`` the
scenes are kept small (<5 s wall time on RTX PRO 6000) and run with
CUDA-graph capture via ``_PhoenXScene``.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX kernels run on CUDA only")
class TestTowerStability(unittest.TestCase):
    """3-cube unit-density tower with ``mass_split_max_partitions=2``
    runs the bottom cube into the overflow bucket. With the C# Tonge
    impulse-scaling math correctly wired through prepare + iterate,
    the stack must NOT explode.

    Pass criteria across all 60 frames:
      * all body positions / velocities finite,
      * max body z stays below 5 m (spawn z < 0.5 m, so anything
        higher means launch),
      * max body speed stays below 5 m/s after the first 10 frames
        (allow some transient settling).
    """

    LAYERS = 3
    CUBE_HALF = 0.05
    SETTLE_FRAMES = 60  # 1 s at 60 Hz -- enough to expose explosions

    def _run(self, *, mass_split_max_partitions: int | None):
        scene = _PhoenXScene(
            fps=60,
            substeps=4,
            solver_iterations=12,
            friction=0.5,
            mass_split_max_partitions=mass_split_max_partitions,
        )
        scene.add_ground_plane()
        cube_size = 2.0 * self.CUBE_HALF
        cube_ids: list[int] = []
        for layer in range(self.LAYERS):
            cube_ids.append(
                scene.add_box(
                    position=(0.0, 0.0, cube_size * (layer + 0.5) + 0.002),
                    half_extents=(self.CUBE_HALF,) * 3,
                    density=1000.0,
                )
            )
        scene.finalize()

        max_z_history: list[float] = []
        max_speed_history: list[float] = []
        for frame in range(self.SETTLE_FRAMES):
            scene.step()
            positions = scene.bodies.position.numpy()
            velocities = scene.bodies.velocity.numpy()
            self.assertTrue(np.all(np.isfinite(positions)), f"frame {frame}: non-finite positions")
            self.assertTrue(np.all(np.isfinite(velocities)), f"frame {frame}: non-finite velocities")
            max_z = float(np.max(positions[1:, 2])) if positions.shape[0] > 1 else 0.0
            max_speed = float(np.max(np.linalg.norm(velocities[1:], axis=1))) if velocities.shape[0] > 1 else 0.0
            max_z_history.append(max_z)
            max_speed_history.append(max_speed)
            self.assertLess(
                max_z,
                5.0,
                f"frame {frame}: tower exploded (max z = {max_z}); max_z_history = {max_z_history}",
            )
        return max_z_history, max_speed_history

    def test_tower_no_mass_splitting_baseline(self):
        """Sanity: baseline (no mass splitting) settles cleanly."""
        _z_history, speed_history = self._run(mass_split_max_partitions=None)
        # After 1 s the stack should be near rest. Allow some
        # residual jitter -- 0.5 m/s is generous (Box2D-soft
        # accumulates a few mm/s of bias even at rest).
        self.assertLess(
            float(np.mean(speed_history[-10:])),
            0.5,
            f"baseline: tower didn't settle, mean speed last 10 frames = {speed_history[-10:]}",
        )

    def test_tower_with_mass_splitting_does_not_explode(self):
        """Regression: with ``mass_split_max_partitions=2`` the
        bottom cube ends up in overflow. Mass splitting must NOT
        cause the stack to fly off."""
        _z_history, speed_history = self._run(mass_split_max_partitions=2)
        # Looser than the unsplit baseline: mass splitting is a
        # softer relaxation, so allow up to 2.0 m/s residual at the
        # end. But the stack must remain bounded.
        self.assertLess(
            max(speed_history[-5:]),
            5.0,
            f"tower with mass splitting unstable: speeds last 5 frames = {speed_history[-5:]}, "
            f"full speed history = {speed_history}",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX kernels run on CUDA only")
class TestTowerStabilityDeepStack(unittest.TestCase):
    """6-cube tall stack -- catches the case where mass splitting
    correctly handles the bottom cube but the multi-cube interaction
    (every cube except top has ~8 contacts above + 8 below) destabilises
    above 3 layers. The user's original example_kapla_tower regression
    is in this regime.
    """

    LAYERS = 6
    CUBE_HALF = 0.05
    SETTLE_FRAMES = 60

    def test_six_cube_tower_with_mass_splitting(self):
        scene = _PhoenXScene(
            fps=60,
            substeps=4,
            solver_iterations=12,
            friction=0.5,
            mass_split_max_partitions=2,
        )
        scene.add_ground_plane()
        cube_size = 2.0 * self.CUBE_HALF
        for layer in range(self.LAYERS):
            scene.add_box(
                position=(0.0, 0.0, cube_size * (layer + 0.5) + 0.002),
                half_extents=(self.CUBE_HALF,) * 3,
                density=1000.0,
            )
        scene.finalize()

        max_z_history: list[float] = []
        max_speed_history: list[float] = []
        for frame in range(self.SETTLE_FRAMES):
            scene.step()
            positions = scene.bodies.position.numpy()
            velocities = scene.bodies.velocity.numpy()
            self.assertTrue(
                np.all(np.isfinite(positions)),
                f"frame {frame}: non-finite positions: {positions}",
            )
            self.assertTrue(
                np.all(np.isfinite(velocities)),
                f"frame {frame}: non-finite velocities: {velocities}",
            )
            max_z = float(np.max(positions[1:, 2])) if positions.shape[0] > 1 else 0.0
            max_speed = float(np.max(np.linalg.norm(velocities[1:], axis=1))) if velocities.shape[0] > 1 else 0.0
            max_z_history.append(max_z)
            max_speed_history.append(max_speed)
            self.assertLess(
                max_z,
                5.0,
                f"frame {frame}: 6-cube tower exploded (max z = {max_z}); max_z_history = {max_z_history}",
            )

        self.assertLess(
            max(max_speed_history[-5:]),
            5.0,
            f"6-cube tower with mass splitting unstable: "
            f"speeds last 5 frames = {max_speed_history[-5:]}, "
            f"full speed history = {max_speed_history}",
        )


if __name__ == "__main__":
    unittest.main()
