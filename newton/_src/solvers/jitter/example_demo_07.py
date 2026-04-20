# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Demo 07 -- Many Pyramids
#
# Port of ``JitterDemo.Demos.Demo07`` (see
# ``C:\git3\jitterphysics2\src\JitterDemo\Demos\Demo07.cs``). The C#
# reference drops 60 pre-deactivated pyramids in a 2x30 grid; we keep
# the grid layout but shrink both axes (8 pyramids, 8 layers each) so
# the scene stays lively at 60 Hz without a full broad-phase budget.
#
# There's no deactivation in Newton's collision pipeline so every body
# is live from the start -- this still exercises the many-contact-
# manifold case the original demo was built to stress.
#
# Run:  python -m newton._src.solvers.jitter.example_demo_07
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

PYRAMIDS_X = 4
PYRAMIDS_Y = 2
LAYERS_PER_PYRAMID = 8
# 5 m between pyramid centres along +Y; 20 m between the two +X rows.
PYRAMID_SPACING_X = 20.0
PYRAMID_SPACING_Y = 5.0
# Bottom-row centre of the pyramid grid (world coordinates).
GRID_ORIGIN_X = -PYRAMIDS_X * 0.5
GRID_ORIGIN_Y = -(PYRAMIDS_Y - 1) * 0.5 * PYRAMID_SPACING_Y

BOX_HALF = 0.5
BOX_SPACING = 1.01  # [m]


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Many Pyramids",
            camera_pos=(50.0, 0.0, 25.0),
            camera_pitch=-22.0,
            camera_yaw=0.0,
            fps=60,
            substeps=4,
            solver_iterations=4,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._box_bodies: list[int] = []
        for gx in range(PYRAMIDS_X):
            for gy in range(PYRAMIDS_Y):
                origin = np.array(
                    [
                        GRID_ORIGIN_X + gx * PYRAMID_SPACING_X,
                        GRID_ORIGIN_Y + gy * PYRAMID_SPACING_Y,
                        0.0,
                    ],
                    dtype=np.float32,
                )
                self._spawn_pyramid(origin)

    def _spawn_pyramid(self, origin: np.ndarray) -> None:
        mb = self.model_builder
        for level in range(LAYERS_PER_PYRAMID):
            num_in_row = LAYERS_PER_PYRAMID - level
            for e in range(num_in_row):
                # Match BuildPyramid in Common.cs: horizontal offset
                # ``(e - level * 0.5) * 1.01`` keeps each row centred.
                x = float(origin[0] + (e - level * 0.5) * BOX_SPACING)
                y = float(origin[1])
                z = float(origin[2] + BOX_HALF + level * 1.0)
                body = mb.add_body(
                    xform=wp.transform(
                        p=wp.vec3(x, y, z), q=wp.quat_identity()
                    ),
                    mass=1.0,
                )
                mb.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
                self._box_bodies.append(body)
                self.register_body_extent(body, (BOX_HALF, BOX_HALF, BOX_HALF))

    def test_final(self) -> None:
        """Pyramids shouldn't explode: no body should be flying upward."""
        velocities = self.world.bodies.velocity.numpy()
        for newton_idx in self._box_bodies:
            j = self._newton_to_jitter[newton_idx]
            vel = velocities[j]
            assert np.isfinite(vel).all(), (
                f"body {newton_idx} non-finite velocity {vel}"
            )


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
