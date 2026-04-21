# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Vertical Stack with Bullet
#
# Port of Box2D's ``Stacking/Vertical Stack`` sample. A single tall
# column of 12 rounded boxes sits on the floor; a heavy sphere is
# fired at it along +X. Good continuous-collision / impact-response
# stress test: the bullet has to hit the column, knock a chunk of it
# over, and the column's lower stack has to hold up under the tumble.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_vertical_stack
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

ROW_COUNT = 12
BOX_HALF = 0.45
LEVEL_HEIGHT = 1.0

BULLET_RADIUS = 0.5
BULLET_START_X = -15.0
BULLET_START_Z = 1.5
BULLET_SPEED = 25.0
BULLET_MASS = 20.0


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Vertical Stack with Bullet",
            camera_pos=(10.0, 18.0, 8.0),
            camera_pitch=-16.0,
            camera_yaw=-70.0,
            fps=60,
            substeps=4,
            solver_iterations=16,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._stack_bodies: list[int] = []
        for row in range(ROW_COUNT):
            z = BOX_HALF + row * LEVEL_HEIGHT + 1.0e-3
            body = mb.add_body(
                xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()),
                mass=1.0,
            )
            mb.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
            self._stack_bodies.append(body)
            self.register_body_extent(body, (BOX_HALF, BOX_HALF, BOX_HALF))

        # Bullet -- heavy sphere flying along +X towards the stack.
        self._bullet_body = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(BULLET_START_X, 0.0, BULLET_START_Z),
                q=wp.quat_identity(),
            ),
            mass=BULLET_MASS,
        )
        mb.add_shape_sphere(self._bullet_body, radius=BULLET_RADIUS)
        self.register_body_extent(
            self._bullet_body, (BULLET_RADIUS, BULLET_RADIUS, BULLET_RADIUS)
        )
        # Remember so we can inject the initial velocity after the
        # Newton<->Jitter mirror is populated.
        self._bullet_initial_velocity = (BULLET_SPEED, 0.0, 0.0)

    def on_jitter_builder_ready(self, builder, newton_to_jitter) -> None:
        # Write the bullet's linear velocity into Newton's qd so
        # ``eval_fk`` + the subsequent Newton->Jitter sync pick it up.
        # ``self.state.body_qd`` layout is (linear, angular); bullet
        # is Newton body index ``self._bullet_body``.
        qd = self.state.body_qd.numpy().copy()
        bullet_idx = int(self._bullet_body)
        qd[bullet_idx][0] = float(self._bullet_initial_velocity[0])
        qd[bullet_idx][1] = float(self._bullet_initial_velocity[1])
        qd[bullet_idx][2] = float(self._bullet_initial_velocity[2])
        self.state.body_qd.assign(qd)

    def test_final(self) -> None:
        # After settle, everything should be at rest or very slow.
        # Stack may have toppled, but no body should have exploded.
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody in [*self._stack_bodies, self._bullet_body]:
            j = self._newton_to_jitter[nbody]
            assert np.isfinite(positions[j]).all(), f"body {nbody} non-finite pos"
            assert np.isfinite(velocities[j]).all(), f"body {nbody} non-finite vel"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
