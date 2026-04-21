# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Demo 11 -- Double Pendulum
#
# Adapted from ``JitterDemo.Demos.Demo11`` (see
# ``C:\git3\jitterphysics2\src\JitterDemo\Demos\Demo11.cs``). The C#
# reference uses a pair of ``DistanceLimit`` constraints (rope-like 1-DoF
# distance constraints); our Jitter port intentionally only ships the
# unified double-ball-socket joint, so we build the pendulum out of two
# :attr:`JointMode.BALL_SOCKET` joints instead.
#
# Topology (identical to the original):
#
#   world anchor (0, 0, 12)
#         |
#         |  ball-socket
#         v
#   small sphere b0 @ (0, 0, 12)
#         |
#         |  ball-socket at b0's position
#         v
#   small sphere b1 @ (0, 0, 13)   (+1 m along +Z)
#
# Jitter is +Y-up; Newton is +Z-up. The C# anchor (0, 8, 0) maps to
# (0, 0, 8) in Newton. Two ball-sockets give a three-dimensional
# double pendulum (the bodies can swing in any direction, not just a
# plane) -- still a faithful "two-link pendulum" demo.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_demo_11
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    WORLD_BODY,
    DemoConfig,
    DemoExample,
)
from newton._src.solvers.jitter.world_builder import JointMode

SPHERE_RADIUS = 0.2
ANCHOR_Z = 8.0


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Double Pendulum",
            camera_pos=(5.0, 5.0, 10.0),
            camera_pitch=-10.0,
            camera_yaw=-45.0,
            fps=60,
            substeps=10,  # matches the C# Demo11 (world.SubstepCount = 10)
            solver_iterations=2,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        # b0: hangs from the world anchor at (0, 0, ANCHOR_Z).
        # Matches the C# b0 initial state -- same +X nudge velocity.
        self._b0 = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, ANCHOR_Z), q=wp.quat_identity()
            ),
            mass=1.0,
        )
        mb.add_shape_sphere(self._b0, radius=SPHERE_RADIUS)
        self.register_body_extent(
            self._b0, (SPHERE_RADIUS, SPHERE_RADIUS, SPHERE_RADIUS)
        )

        # b1: hangs 1 m below b0 along +Z. Same +Y nudge as C#.
        self._b1 = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, ANCHOR_Z - 1.0), q=wp.quat_identity()
            ),
            mass=1.0,
        )
        mb.add_shape_sphere(self._b1, radius=SPHERE_RADIUS)
        self.register_body_extent(
            self._b1, (SPHERE_RADIUS, SPHERE_RADIUS, SPHERE_RADIUS)
        )

        # First link: world anchor <-> b0 at b0's initial position.
        self.add_joint(
            body1=WORLD_BODY,
            body2=self._b0,
            anchor1=(0.0, 0.0, ANCHOR_Z),
            mode=JointMode.BALL_SOCKET,
        )
        # Second link: b0 <-> b1 at b0's initial position (so b1 hangs
        # 1 m below b0 along +Z at t=0).
        self.add_joint(
            body1=self._b0,
            body2=self._b1,
            anchor1=(0.0, 0.0, ANCHOR_Z),
            mode=JointMode.BALL_SOCKET,
        )

    def finish_setup(self) -> None:
        super().finish_setup()
        # Give the pendulum the same tiny initial nudge the C# demo
        # uses (Velocity(0.01f, 0, 0) on b0 and (0, 0, 0.01f) on b1
        # after +Y/+Z swap).
        velocities = self.world.bodies.velocity.numpy().copy()
        velocities[self._newton_to_jitter[self._b0]] = (0.01, 0.0, 0.0)
        velocities[self._newton_to_jitter[self._b1]] = (0.0, 0.01, 0.0)
        self.world.bodies.velocity.assign(np.asarray(velocities, dtype=np.float32))
        self._sync_jitter_to_newton()

    def test_final(self) -> None:
        """The pendulum shouldn't drift off -- both bodies must stay
        within 2 m of the world anchor (well inside the maximal 2 m
        two-rung extension).
        """
        max_radius = 2.2  # [m]
        anchor = np.array([0.0, 0.0, ANCHOR_Z], dtype=np.float32)
        for body in (self._b0, self._b1):
            pos, _vel = self.jitter_body_state(body)
            dist = float(np.linalg.norm(pos - anchor))
            assert dist < max_radius, (
                f"pendulum body {body} drifted to {dist:.2f} m from anchor"
            )


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
