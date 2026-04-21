# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Cliff
#
# Port of Box2D's ``Stacking/Cliff`` sample, generalised to 3D. A
# wedge-shaped ramp launches a tumble of dynamic boxes off an edge;
# the boxes then pile up against a back wall. Tests friction on an
# inclined surface + contact stability at an edge transition + the
# compound "ramp + wall" static geometry.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_cliff
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

RAMP_LENGTH = 12.0
RAMP_WIDTH = 6.0
RAMP_ANGLE_DEG = 18.0
# Number of boxes tumbling off the ramp.
BOX_COUNT = 20
BOX_HALF = 0.3


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Cliff",
            camera_pos=(12.0, 14.0, 6.0),
            camera_pitch=-18.0,
            camera_yaw=-55.0,
            fps=60,
            substeps=4,
            solver_iterations=12,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        # ---- Inclined ramp (static body rotated about +Y by RAMP_ANGLE) ----
        angle = math.radians(RAMP_ANGLE_DEG)
        half_len = 0.5 * RAMP_LENGTH
        half_wid = 0.5 * RAMP_WIDTH
        ramp_half_thick = 0.2
        # Pivot the ramp around its low-edge corner (x=-half_len, z=0);
        # the high edge is lifted to half_len*sin(angle).
        # Place the ramp body so the low corner sits at x = ramp_end_x
        # where the drop happens.
        ramp_end_x = 4.0
        ramp_center_x = ramp_end_x - half_len * math.cos(angle)
        ramp_center_z = half_len * math.sin(angle) + ramp_half_thick * math.cos(angle)
        # Quaternion for rotation about +Y by +angle.
        sin_h = math.sin(angle * 0.5)
        cos_h = math.cos(angle * 0.5)
        ramp_q = wp.quat(0.0, sin_h, 0.0, cos_h)
        self._ramp_body = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(ramp_center_x, 0.0, ramp_center_z), q=ramp_q
            ),
            mass=0.0,  # static
        )
        mb.add_shape_box(
            self._ramp_body,
            hx=half_len,
            hy=half_wid,
            hz=ramp_half_thick,
        )

        # ---- Back wall beyond the cliff edge ----
        wall_height = 3.0
        wall_x = ramp_end_x + 6.0
        self._wall_body = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(wall_x, 0.0, wall_height * 0.5),
                q=wp.quat_identity(),
            ),
            mass=0.0,
        )
        mb.add_shape_box(
            self._wall_body,
            hx=0.2,
            hy=half_wid,
            hz=wall_height * 0.5,
        )

        # ---- Dynamic boxes lined up on the high end of the ramp ----
        # Start at x = ramp_end_x - RAMP_LENGTH, stagger up-slope.
        self._box_bodies: list[int] = []
        for i in range(BOX_COUNT):
            # Along-ramp position (u) from 0 (top of ramp) to near bottom.
            u = 0.5 + i * (2.0 * BOX_HALF + 0.1)
            if u >= RAMP_LENGTH - 0.5:
                break
            # Place in world space: walk down the ramp from its top.
            top_corner_x = ramp_end_x - RAMP_LENGTH * math.cos(angle)
            top_corner_z = RAMP_LENGTH * math.sin(angle)
            # Move distance u down the slope towards ramp_end_x.
            x = top_corner_x + u * math.cos(angle)
            z = top_corner_z - u * math.sin(angle) + BOX_HALF + 0.01
            body = mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(x, 0.0, z), q=wp.quat_identity()
                ),
                mass=0.5,
            )
            mb.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
            self._box_bodies.append(body)
            self.register_body_extent(body, (BOX_HALF, BOX_HALF, BOX_HALF))

    def test_final(self) -> None:
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody in self._box_bodies:
            j = self._newton_to_jitter[nbody]
            assert np.isfinite(positions[j]).all(), f"body {nbody} non-finite"
            assert np.isfinite(velocities[j]).all(), f"body {nbody} non-finite"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
