# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Double Domino
#
# Port of Box2D's ``Stacking/Double Domino`` sample. Two parallel
# rows of tall thin domino boxes standing on the ground. The first
# domino of each row is given a small initial kick so it topples
# into its neighbour, which topples into the next, producing a
# chain-reaction wave propagating along both rows in parallel.
#
# Stresses: many sliding / toppling contacts simultaneously, stable
# tall-body stability, and impulse propagation along an otherwise
# balanced sequence.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_double_domino
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

DOMINO_COUNT = 25
DOMINO_HX = 0.08   # thickness along the row
DOMINO_HY = 0.5    # depth (perpendicular to the row)
DOMINO_HZ = 1.0    # height

# Spacing between consecutive dominoes along +X.
SPACING = DOMINO_HX * 2.0 + 0.6
# Separation between the two parallel rows along +Y.
ROW_SEPARATION = 2.5

# Initial horizontal nudge for the first domino of each row.
KICK_SPEED = 1.2


class Example(DemoExample):
    def __init__(self, viewer, args):
        span = DOMINO_COUNT * SPACING
        cfg = DemoConfig(
            title="Double Domino",
            camera_pos=(span * 0.2, span * 0.6, DOMINO_HZ * 3.0),
            camera_pitch=-14.0,
            camera_yaw=-80.0,
            fps=60,
            substeps=4,
            solver_iterations=12,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._kick_bodies: list[int] = []
        self._domino_bodies: list[int] = []

        x_origin = -0.5 * (DOMINO_COUNT - 1) * SPACING
        for row_sign in (+1, -1):
            y = row_sign * 0.5 * ROW_SEPARATION
            for i in range(DOMINO_COUNT):
                x = x_origin + i * SPACING
                z = DOMINO_HZ + 1.0e-3
                body = mb.add_body(
                    xform=wp.transform(
                        p=wp.vec3(x, y, z), q=wp.quat_identity()
                    ),
                    mass=1.0,
                )
                mb.add_shape_box(body, hx=DOMINO_HX, hy=DOMINO_HY, hz=DOMINO_HZ)
                self._domino_bodies.append(body)
                self.register_body_extent(body, (DOMINO_HX, DOMINO_HY, DOMINO_HZ))
                if i == 0:
                    self._kick_bodies.append(body)

    def on_jitter_builder_ready(self, builder, newton_to_jitter) -> None:
        qd = self.state.body_qd.numpy().copy()
        for body in self._kick_bodies:
            qd[int(body)][0] = float(KICK_SPEED)  # +X linear velocity
        self.state.body_qd.assign(qd)

    def test_final(self) -> None:
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody in self._domino_bodies:
            j = self._newton_to_jitter[nbody]
            assert np.isfinite(positions[j]).all(), f"body {nbody} non-finite"
            assert np.isfinite(velocities[j]).all(), f"body {nbody} non-finite"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
