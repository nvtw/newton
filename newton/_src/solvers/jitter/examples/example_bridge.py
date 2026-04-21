# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Bridge
#
# Port of Box2D's ``Joints/Bridge`` sample. A chain of plank boxes is
# connected end-to-end by revolute joints and anchored at both ends
# to the world, forming a rope-bridge. Gravity makes it sag; a pile
# of cubes is dropped on top to drive large displacements through
# the chain and test the joint path under load.
#
# Stresses:
#   * long chain of revolute joints with sympathetic resonances,
#   * joint + contact coupling (cubes push planks, planks fight back
#     through the chain),
#   * Jacobi-within-slot convergence on a joint-heavy coloured graph.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_bridge
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
    WORLD_BODY,
)
from newton._src.solvers.jitter.world_builder import JointMode

PLANK_COUNT = 30
PLANK_HALF_LEN = 0.5
PLANK_HALF_THICK = 0.125
# Vertical position of the bridge anchors.
BRIDGE_Z = 5.0

DROP_COUNT = 5
DROP_HALF = 0.3
DROP_START_Z = 10.0


class Example(DemoExample):
    def __init__(self, viewer, args):
        span = PLANK_COUNT * 2.0 * PLANK_HALF_LEN
        cfg = DemoConfig(
            title="Bridge",
            camera_pos=(span * 0.2, span * 0.8, BRIDGE_Z + 4.0),
            camera_pitch=-14.0,
            camera_yaw=-80.0,
            fps=60,
            substeps=4,
            solver_iterations=16,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        # Lay out PLANK_COUNT planks end-to-end along +X, centred on 0.
        # Each plank is 2 * PLANK_HALF_LEN long; the pivot between plank
        # i and plank i-1 sits at x = x_base + i * 2*PLANK_HALF_LEN.
        x_base = -PLANK_COUNT * PLANK_HALF_LEN

        self._plank_bodies: list[int] = []
        prev_body = WORLD_BODY
        for i in range(PLANK_COUNT):
            # Plank centre.
            x = x_base + PLANK_HALF_LEN + i * 2.0 * PLANK_HALF_LEN
            body = mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(x, 0.0, BRIDGE_Z), q=wp.quat_identity()
                ),
                mass=0.2,
            )
            mb.add_shape_box(
                body,
                hx=PLANK_HALF_LEN,
                hy=PLANK_HALF_THICK,
                hz=PLANK_HALF_THICK,
            )
            self._plank_bodies.append(body)
            self.register_body_extent(
                body, (PLANK_HALF_LEN, PLANK_HALF_THICK, PLANK_HALF_THICK)
            )

            # Hinge between this plank and the previous link.
            pivot_x = x - PLANK_HALF_LEN
            self.add_joint(
                body1=prev_body,
                body2=body,
                anchor1=(pivot_x, 0.0, BRIDGE_Z),
                anchor2=(pivot_x, 1.0, BRIDGE_Z),  # axis = +Y
                mode=JointMode.REVOLUTE,
            )

            # Suppress contacts between neighbouring planks -- without
            # this their adjacent faces rattle against each other at
            # the joint and the bridge jitters visibly. Anchoring at
            # their shared edge is enough; the joint enforces the
            # kinematic constraint.
            if prev_body != WORLD_BODY:
                self.add_collision_filter_pair(prev_body, body)

            prev_body = body

        # Close the bridge with a hinge between the final plank and
        # the world anchor on the +X side.
        last_plank = self._plank_bodies[-1]
        final_pivot_x = x_base + PLANK_COUNT * 2.0 * PLANK_HALF_LEN
        self.add_joint(
            body1=last_plank,
            body2=WORLD_BODY,
            anchor1=(final_pivot_x, 0.0, BRIDGE_Z),
            anchor2=(final_pivot_x, 1.0, BRIDGE_Z),
            mode=JointMode.REVOLUTE,
        )

        # ---- A pile of cubes above the bridge ----
        self._drop_bodies: list[int] = []
        d = 2.0 * DROP_HALF + 0.05
        for i in range(DROP_COUNT):
            x = -DROP_COUNT * 0.5 * d + i * d
            body = mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(x, 0.0, DROP_START_Z), q=wp.quat_identity()
                ),
                mass=1.0,
            )
            mb.add_shape_box(body, hx=DROP_HALF, hy=DROP_HALF, hz=DROP_HALF)
            self._drop_bodies.append(body)
            self.register_body_extent(body, (DROP_HALF, DROP_HALF, DROP_HALF))

    def test_final(self) -> None:
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody in [*self._plank_bodies, *self._drop_bodies]:
            j = self._newton_to_jitter[nbody]
            assert np.isfinite(positions[j]).all(), f"body {nbody} non-finite"
            assert np.isfinite(velocities[j]).all(), f"body {nbody} non-finite"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
