# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Tumbler
#
# Port of Box2D's ``Benchmark/Tumbler`` sample. A big hollow box
# (four plank shapes attached to one body) spins about a vertical
# revolute axis driven by a velocity motor. A cube of debris is
# dropped inside and tumbles as the box rotates. Stresses:
#
#   * a motorised revolute joint (rotating container),
#   * many dynamic-vs-dynamic contacts against the fast-moving
#     inner walls (the plank shapes sweep through the debris every
#     revolution),
#   * warm-starting under continuously changing contact manifolds.
#
# In Box2D the container is 2D (a square frame) and the axis runs
# out of the screen. Ported to 3D we make it a 4-plank square frame
# spinning around +Z so the debris stays inside and grinds against
# the rotating walls.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_tumbler
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
    WORLD_BODY,
)
from newton._src.solvers.jitter.world_builder import DriveMode, JointMode

# Half-extents of the outer container (a hollow box made from four
# thin planks). Inside clearance ~2 * INNER_HALF.
INNER_HALF = 4.0
PLANK_HALF_THICK = 0.2

# Debris grid.
DEBRIS_SIDE = 6
DEBRIS_HALF = 0.2

# Motor speed (rad/s about world +Z).
MOTOR_SPEED = math.radians(45.0)
MAX_MOTOR_TORQUE = 1.0e8


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Tumbler",
            camera_pos=(INNER_HALF * 3.0, INNER_HALF * 3.0, INNER_HALF * 1.5),
            camera_pitch=-20.0,
            camera_yaw=-45.0,
            fps=60,
            substeps=6,
            solver_iterations=16,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        # ---- Container body ----
        # Enough density that the container has significant inertia;
        # otherwise the motor spins a near-massless lattice and the
        # debris basically slides through.
        container_z = INNER_HALF + 1.0
        self._container_body = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, container_z), q=wp.quat_identity()
            ),
            mass=50.0,
        )
        # Four plank shapes at +/- X and +/- Y faces of the container.
        # ``xform`` here is the shape's local frame within the body.
        wall_offset = INNER_HALF
        wall_len = INNER_HALF + PLANK_HALF_THICK
        wall_height = INNER_HALF
        # +X wall (thin along X, long along Y + Z)
        mb.add_shape_box(
            self._container_body,
            hx=PLANK_HALF_THICK,
            hy=wall_len,
            hz=wall_height,
            xform=wp.transform(
                p=wp.vec3(+wall_offset, 0.0, 0.0), q=wp.quat_identity()
            ),
        )
        # -X wall
        mb.add_shape_box(
            self._container_body,
            hx=PLANK_HALF_THICK,
            hy=wall_len,
            hz=wall_height,
            xform=wp.transform(
                p=wp.vec3(-wall_offset, 0.0, 0.0), q=wp.quat_identity()
            ),
        )
        # +Y wall
        mb.add_shape_box(
            self._container_body,
            hx=wall_len,
            hy=PLANK_HALF_THICK,
            hz=wall_height,
            xform=wp.transform(
                p=wp.vec3(0.0, +wall_offset, 0.0), q=wp.quat_identity()
            ),
        )
        # -Y wall
        mb.add_shape_box(
            self._container_body,
            hx=wall_len,
            hy=PLANK_HALF_THICK,
            hz=wall_height,
            xform=wp.transform(
                p=wp.vec3(0.0, -wall_offset, 0.0), q=wp.quat_identity()
            ),
        )
        # Floor of the container so debris stays inside.
        mb.add_shape_box(
            self._container_body,
            hx=wall_len,
            hy=wall_len,
            hz=PLANK_HALF_THICK,
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, -wall_height), q=wp.quat_identity()
            ),
        )
        self.register_body_extent(
            self._container_body, (wall_len, wall_len, wall_height)
        )

        # Vertical revolute joint anchored at the container COM,
        # hinge axis = world +Z. ``add_joint`` takes two world-space
        # anchors; the line between them is the hinge axis.
        anchor = (0.0, 0.0, container_z)
        self.add_joint(
            body1=WORLD_BODY,
            body2=self._container_body,
            anchor1=anchor,
            anchor2=(anchor[0], anchor[1], anchor[2] + 1.0),
            mode=JointMode.REVOLUTE,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=MOTOR_SPEED,
            max_force_drive=MAX_MOTOR_TORQUE,
            stiffness_drive=0.0,
            damping_drive=MAX_MOTOR_TORQUE,
        )

        # ---- Debris cubes inside the container ----
        self._debris_bodies: list[int] = []
        d = 2.0 * DEBRIS_HALF + 0.05
        origin = -0.5 * (DEBRIS_SIDE - 1) * d
        for i in range(DEBRIS_SIDE):
            for j in range(DEBRIS_SIDE):
                for k in range(DEBRIS_SIDE):
                    pos = wp.vec3(
                        origin + i * d,
                        origin + j * d,
                        container_z + origin + k * d,
                    )
                    body = mb.add_body(
                        xform=wp.transform(p=pos, q=wp.quat_identity()),
                        mass=0.1,
                    )
                    mb.add_shape_box(
                        body, hx=DEBRIS_HALF, hy=DEBRIS_HALF, hz=DEBRIS_HALF
                    )
                    self._debris_bodies.append(body)
                    self.register_body_extent(
                        body, (DEBRIS_HALF, DEBRIS_HALF, DEBRIS_HALF)
                    )

    def test_final(self) -> None:
        # Stress test -- just check finite.
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody in [self._container_body, *self._debris_bodies]:
            j = self._newton_to_jitter[nbody]
            assert np.isfinite(positions[j]).all(), f"body {nbody} non-finite"
            assert np.isfinite(velocities[j]).all(), f"body {nbody} non-finite"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
