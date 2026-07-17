# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kapla contact-regression tests for PhoenX.

The reduced fixture uses the bottom slice of the real Kapla tower data
so it keeps the same plank dimensions, contact gap, and initial
overlaps as :mod:`example_kapla_tower`, while staying small enough for
the unit-test suite.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.examples import example_kapla_square_tower, example_kapla_tower
from newton._src.solvers.phoenx.examples.example_kapla_tower import BRICK_DENSITY, GLOBAL_SCALING, GROUND_HEIGHT
from newton._src.solvers.phoenx.examples.kapla_tower_data import BRICK_FULL_EXTENTS, ORIENTATIONS, POSITIONS
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene
from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX Kapla regression tests require CUDA graph capture")
class TestPhoenXKaplaPrimitiveContacts(unittest.TestCase):
    """Primitive Kapla contacts must settle after graph-captured stepping."""

    def test_tower_releases_warmup_damping(self) -> None:
        """The tower must run undamped after its overlap warmup."""
        example = example_kapla_tower.Example(
            ViewerNull(),
            SimpleNamespace(solver="classic", max_colors=10),
        )
        for _ in range(example.WARMUP_FRAMES + 1):
            example.step()

        self.assertEqual(example.world.get_global_linear_damping(), 0.0)
        self.assertEqual(example.world.get_global_angular_damping(), 0.0)

    def test_reduced_kapla_slice_settles_after_speculative_contacts(self) -> None:
        """A real-data Kapla slice must not keep moving after settle.

        This fails when the SDF stale-anchor start-gap barrier is also
        applied to primitive contacts: positive-gap primitive rows keep
        losing warm-start/friction state and the slice is still moving
        at order-1 m/s after 150 frames.
        """

        brick_ids = np.nonzero(POSITIONS[:, 2] <= 0.7)[0]
        scene = _PhoenXScene(
            fps=120,
            substeps=6,
            solver_iterations=10,
            velocity_iterations=1,
            friction=0.5,
            step_layout="single_world",
            mass_splitting=True,
            colored_contact_headers=True,
            colored_contact_rows=True,
            max_thread_blocks=384,
            max_colored_partitions=8,
            mass_splitting_batch_size=1,
        )
        scene.mb.default_shape_cfg.gap = 0.01
        scene.mb.add_ground_plane(height=GROUND_HEIGHT)

        hx = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[0]
        hy = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[1]
        hz = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[2]
        for brick_id in brick_ids:
            quat = ORIENTATIONS[brick_id].astype(np.float32)
            quat = quat / max(float(np.linalg.norm(quat)), 1.0e-12)
            pos = (POSITIONS[brick_id] * GLOBAL_SCALING).astype(np.float32)
            scene.add_box(
                tuple(float(x) for x in pos),
                (hx, hy, hz),
                orientation=tuple(float(x) for x in quat),
                density=BRICK_DENSITY,
            )

        scene.finalize()

        for _ in range(300):
            scene.step()
        self.assertIsNotNone(scene._graph)

        velocities = scene.bodies.velocity.numpy()[1:]
        angular_velocities = scene.bodies.angular_velocity.numpy()[1:]
        positions = scene.bodies.position.numpy()[1:]

        max_speed = float(np.linalg.norm(velocities, axis=1).max())
        max_angular_speed = float(np.linalg.norm(angular_velocities, axis=1).max())
        self.assertTrue(np.isfinite(positions).all())
        self.assertLess(max_speed, 0.2, f"Kapla slice did not settle: max |v|={max_speed:.3f} m/s")
        self.assertLess(
            max_angular_speed,
            5.0,
            f"Kapla slice did not settle: max |w|={max_angular_speed:.3f} rad/s",
        )

    def test_thrown_square_tower_does_not_gain_unbounded_spin(self) -> None:
        """An off-axis throw must not inject energy into anisotropic planks."""
        example = example_kapla_square_tower.Example(
            ViewerNull(),
            SimpleNamespace(grid_side=1, show_contacts=False),
        )
        target = example._tower_plank_newton_ids[0][len(example._tower_plank_newton_ids[0]) // 2] + 1
        peak_angular_speed = 0.0
        peak_radius = 0.0

        for frame in range(240):
            if 100 <= frame < 124:
                force = example.bodies.force.numpy()
                torque = example.bodies.torque.numpy()
                force[target] = (150.0, 55.5, 0.0)
                torque[target] = (0.0, 15.0, 7.5)
                example.bodies.force.assign(force)
                example.bodies.torque.assign(torque)
            example.step()
            if frame % 10 == 0 or frame == 239:
                positions = example.bodies.position.numpy()[1:]
                angular_velocities = example.bodies.angular_velocity.numpy()[1:]
                mask = np.arange(len(positions)) != target - 1
                self.assertTrue(np.isfinite(positions).all())
                peak_angular_speed = max(
                    peak_angular_speed,
                    float(np.linalg.norm(angular_velocities[mask], axis=1).max()),
                )
                peak_radius = max(peak_radius, float(np.linalg.norm(positions[mask, :2], axis=1).max()))

        self.assertLess(peak_angular_speed, 500.0)
        self.assertLess(peak_radius, 12.0)


if __name__ == "__main__":
    unittest.main()
