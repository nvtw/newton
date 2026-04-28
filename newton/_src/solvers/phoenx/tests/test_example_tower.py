# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression test mirroring
:mod:`newton._src.solvers.phoenx.examples.example_tower`.

Builds the exact 40-layer / 32-planks-per-ring circular stack the
example uses (same half-extents, same alternating-orientation
pattern, same solver settings) and asserts that after a short settle
no plank has dropped more than a fraction of a plank height from its
initial layer. Designed to catch solver regressions that let resting
contacts leak through each other -- ``example_tower`` is PhoenX's
canonical jitter benchmark and the first scene to break when
cross-colour PGS feedback is weakened.

Runs on CUDA only (same constraint as the rest of the PhoenX suite).
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene

# ---- Tower geometry (must stay in sync with example_tower.py) ----
_TOWER_HEIGHT_LAYERS = 40
_BOXES_PER_RING = 32
_PLANK_HX = 1.5  # tangential
_PLANK_HY = 0.1  # radial wall thickness
_PLANK_HZ = 0.5  # vertical half-extent
_RING_RADIUS = 19.5
_HALF_ROTATION_STEP = 2.0 * math.pi / 64.0
_FULL_ROTATION_STEP = 2.0 * _HALF_ROTATION_STEP
_PLANK_DENSITY = 1000.0


def _spawn_tower_plank_transforms() -> list[tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    """Reproduce ``example_tower.Example._build_scene``'s plank layout.

    Returns a list of ``(position, quat_xyzw)`` pairs in the same order
    the example adds them, so a body-index-matched settlement check
    can compare back against this layout.
    """
    transforms: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]] = []
    orientation_rad = 0.0
    for e in range(_TOWER_HEIGHT_LAYERS):
        orientation_rad += _HALF_ROTATION_STEP
        for _ in range(_BOXES_PER_RING):
            cos_o = math.cos(orientation_rad)
            sin_o = math.sin(orientation_rad)
            local_y = _RING_RADIUS
            local_z = 0.5 + e
            world_x = -sin_o * local_y
            world_y = cos_o * local_y
            world_z = local_z
            # ``wp.quat_from_axis_angle`` around +Z -> quat (0, 0, sin(a/2), cos(a/2))
            half = orientation_rad * 0.5
            quat = (0.0, 0.0, math.sin(half), math.cos(half))
            transforms.append(((world_x, world_y, world_z), quat))
            orientation_rad += _FULL_ROTATION_STEP
    return transforms


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX tower test requires CUDA")
class TestExampleTowerNothingDrops(unittest.TestCase):
    """Full-scale ``example_tower`` regression: no plank falls.

    Simulates the 1280-plank circular tower through 60 frames at
    60 Hz (1 s) with the same solver settings the example uses
    (``substeps = 20``, ``solver_iterations = 3``,
    ``velocity_iterations = 1``, the new minimum after the soft-PD
    damping split). Asserts every plank stays within half a plank
    height of its initial layer centre.

    The tolerance is deliberately generous (``0.5 * PLANK_HZ`` = 25 cm)
    -- a settled circular stack compresses a couple of mm under its
    own weight. A full layer drop (which is what the regression
    produces) is 1.0 m, so 25 cm is well below the failure
    signature but comfortably above the resting compression.
    """

    # Use the example's cadence.
    SIM_SUBSTEPS = 20
    SOLVER_ITERATIONS = 3
    FPS = 60
    SETTLE_FRAMES = 60  # 1 second at 60 Hz

    def test_tower_nothing_drops(self) -> None:
        scene = _PhoenXScene(
            fps=self.FPS,
            substeps=self.SIM_SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            velocity_iterations=1,
            friction=0.5,
        )
        scene.add_ground_plane()

        transforms = _spawn_tower_plank_transforms()
        plank_ids: list[int] = []
        initial_z: list[float] = []
        for pos, quat in transforms:
            body = scene.add_box(
                position=pos,
                half_extents=(_PLANK_HX, _PLANK_HY, _PLANK_HZ),
                orientation=quat,
                density=_PLANK_DENSITY,
            )
            plank_ids.append(body)
            initial_z.append(pos[2])

        scene.finalize()

        for _ in range(self.SETTLE_FRAMES):
            scene.step()

        positions = scene.bodies.position.numpy()

        # Per-layer expected z (half a plank above layer integer z).
        drop_tol = 0.5 * _PLANK_HZ  # 0.25 m -- well below 1.0 m layer drop
        max_drop = 0.0
        worst_body = -1
        worst_layer = -1
        for i, (body, z0) in enumerate(zip(plank_ids, initial_z, strict=False)):
            slot = body + 1  # slot 0 = world anchor
            z = float(positions[slot, 2])
            drop = z0 - z
            if drop > max_drop:
                max_drop = drop
                worst_body = body
                worst_layer = i // _BOXES_PER_RING
            # Also catch NaN / fly-away explosion.
            self.assertTrue(
                np.isfinite(positions[slot]).all(),
                f"plank {body} (layer {i // _BOXES_PER_RING}) has non-finite position",
            )

        self.assertLess(
            max_drop,
            drop_tol,
            f"tower dropped: worst plank {worst_body} (layer {worst_layer}) "
            f"fell {max_drop:.3f} m (tol={drop_tol:.3f} m). "
            f"Initial layer z was ~{initial_z[worst_body]:.3f} m, "
            f"observed z {float(positions[worst_body + 1, 2]):.3f} m.",
        )

        # Secondary sanity: nothing flying. The example's own
        # ``test_final`` uses RING_RADIUS * 3 as envelope; keep that
        # same envelope here so this test also catches the
        # blow-up-and-eject class of failure.
        envelope = _RING_RADIUS * 3.0
        for body in plank_ids:
            pos = positions[body + 1]
            r_xy = float(math.hypot(pos[0], pos[1]))
            self.assertLess(
                r_xy,
                envelope,
                f"plank {body} flew outside tower envelope (r_xy={r_xy:.2f}, tol={envelope:.2f})",
            )


if __name__ == "__main__":
    unittest.main()
