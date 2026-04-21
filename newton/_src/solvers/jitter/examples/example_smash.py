# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Smash
#
# Port of Box2D's ``Benchmark/Smash`` scene. A fast, heavy box flies
# into a dense grid of small boxes under zero gravity. The 2D reference
# spawns a ``120 x 80`` grid; the 3D port uses a ``20 x 20 x 20`` cube
# so every column of the grid is seen by the ballistic box. Stresses
# the solver's ability to propagate impulse through a many-body field
# of contacts in a single frame.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_smash
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

# Grid of small boxes (scale down from Box2D's 120x80 = 9600 to keep
# the scene interactive on a single GPU).
GRID_X = 15
GRID_Y = 15
GRID_Z = 15
SMALL_HALF = 0.2

RAM_HALF = 2.0
RAM_MASS = 100.0
RAM_SPEED = 30.0


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Smash (zero-gravity impact)",
            camera_pos=(10.0, 35.0, 8.0),
            camera_pitch=-14.0,
            camera_yaw=-80.0,
            fps=60,
            substeps=4,
            solver_iterations=12,
            gravity=(0.0, 0.0, 0.0),
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder

        # Grid of small cubes centred around the world origin.
        d = 2.0 * SMALL_HALF + 0.01  # tiny gap so broadphase doesn't penetrate
        self._small_bodies: list[int] = []
        x_origin = -0.5 * (GRID_X - 1) * d
        y_origin = -0.5 * (GRID_Y - 1) * d
        z_origin = -0.5 * (GRID_Z - 1) * d
        for i in range(GRID_X):
            for j in range(GRID_Y):
                for k in range(GRID_Z):
                    pos = wp.vec3(
                        x_origin + i * d,
                        y_origin + j * d,
                        z_origin + k * d,
                    )
                    body = mb.add_body(
                        xform=wp.transform(p=pos, q=wp.quat_identity()),
                        mass=1.0,
                    )
                    mb.add_shape_box(
                        body, hx=SMALL_HALF, hy=SMALL_HALF, hz=SMALL_HALF
                    )
                    self._small_bodies.append(body)
                    self.register_body_extent(
                        body, (SMALL_HALF, SMALL_HALF, SMALL_HALF)
                    )

        # Ram -- a heavy box far away along -X heading along +X.
        ram_x = x_origin - 10.0
        self._ram_body = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(ram_x, 0.0, 0.0), q=wp.quat_identity()
            ),
            mass=RAM_MASS,
        )
        mb.add_shape_box(
            self._ram_body, hx=RAM_HALF, hy=RAM_HALF, hz=RAM_HALF
        )
        self.register_body_extent(
            self._ram_body, (RAM_HALF, RAM_HALF, RAM_HALF)
        )

    def on_jitter_builder_ready(self, builder, newton_to_jitter) -> None:
        qd = self.state.body_qd.numpy().copy()
        qd[int(self._ram_body)][0] = float(RAM_SPEED)
        self.state.body_qd.assign(qd)

    def test_final(self) -> None:
        # Smash is a stress test -- just check no NaNs / no explosions.
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody in [*self._small_bodies, self._ram_body]:
            j = self._newton_to_jitter[nbody]
            assert np.isfinite(positions[j]).all(), f"body {nbody} non-finite"
            assert np.isfinite(velocities[j]).all(), f"body {nbody} non-finite"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
