# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Tilted Stack
#
# Port of Box2D's ``Stacking/Tilted Stack`` sample. A small horizontal
# offset between levels produces a deliberately tilted column that a
# correct solver can still keep from toppling. Ten towers, ten levels
# each, every level shifted by ``offset`` along +X from the one below.
# Good single-stack / correlated-error stress test for friction +
# positional bias.
#
# Box2D 2D scene lives in the XY plane with gravity along -Y. Ported
# to 3D: towers stand along +Z, the tilt lives in the XZ plane, and
# rows of towers extend along +Y so the columns don't interact.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_tilted_stack
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

COLUMNS = 10
ROWS = 10
BOX_HALF = 0.45
# Per-level tilt along +X (matches Box2D's 0.2 m offset).
TILT_OFFSET = 0.2
# Pitch between tower base centres along +Y.
TOWER_PITCH = 5.0
# Initial vertical spacing -- tight but non-penetrating.
LEVEL_HEIGHT = 1.0


class Example(DemoExample):
    def __init__(self, viewer, args):
        span_y = COLUMNS * TOWER_PITCH
        span_z = ROWS * LEVEL_HEIGHT
        cfg = DemoConfig(
            title="Tilted Stack",
            camera_pos=(span_y * 0.6, span_y * 1.2, span_z * 0.8),
            camera_pitch=-18.0,
            camera_yaw=-60.0,
            fps=60,
            substeps=4,
            solver_iterations=12,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._box_bodies: list[int] = []

        y_origin = -0.5 * (COLUMNS - 1) * TOWER_PITCH
        for col in range(COLUMNS):
            y = y_origin + col * TOWER_PITCH
            for row in range(ROWS):
                x = TILT_OFFSET * row
                z = BOX_HALF + row * LEVEL_HEIGHT + 1.0e-3
                body = mb.add_body(
                    xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()),
                    mass=1.0,
                )
                mb.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
                self._box_bodies.append(body)
                self.register_body_extent(body, (BOX_HALF, BOX_HALF, BOX_HALF))

    def test_final(self) -> None:
        """After settle: every cube finite and not running away."""
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody in self._box_bodies:
            j = self._newton_to_jitter[nbody]
            assert np.isfinite(positions[j]).all(), f"body {nbody} non-finite"
            assert np.isfinite(velocities[j]).all(), f"body {nbody} non-finite vel"
            # Tilted stacks can lean significantly; loose but bounded.
            assert float(np.linalg.norm(velocities[j])) < 2.0


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
