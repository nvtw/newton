# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Joint Grid
#
# Port of Box2D's ``Benchmark/Joint Grid`` scene
# (``shared/benchmarks.c::CreateJointGrid``). An ``N x N`` grid of
# spheres connected to their four Manhattan-neighbours by revolute
# joints. A short central row is pinned to the world; the rest of
# the grid hangs under gravity and swings like a chainmail curtain.
#
# Excellent joint-dominated stress test: the solver sees ``O(N^2)``
# revolute constraints arranged in a locked, cyclic mesh. The graph
# colouring's job is to partition these into independent sub-islands
# at every iteration.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_joint_grid
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

# Grid size. Box2D's benchmark uses 100; scale down so the scene stays
# responsive (100x100 = 10000 bodies is too heavy for interactive
# exploration but still fine as a benchmark knob).
N = 20
SPHERE_RADIUS = 0.4
# Pitch between grid cells along +X and +Z. 1.0 m yields a taut mesh
# with no slack: the revolute anchors sit 1 m apart, joints rest length 0.
GRID_PITCH = 1.0


class Example(DemoExample):
    def __init__(self, viewer, args):
        span = N * GRID_PITCH
        cfg = DemoConfig(
            title=f"Joint Grid ({N}x{N})",
            camera_pos=(span * 1.2, span * 0.8, 0.0),
            camera_pitch=-10.0,
            camera_yaw=-30.0,
            fps=60,
            substeps=4,
            solver_iterations=16,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder

        # Build the grid in the XZ plane (so gravity along -Z makes the
        # hanging mesh drape naturally downward).
        self._bodies: list[list[int]] = [[-1] * N for _ in range(N)]
        # Top row centred on y=0, z=0 for the static anchors.
        z_top = 0.0

        for k in range(N):
            for i in range(N):
                # Static if near the top row at column 0.
                is_pinned = (k >= N // 2 - 2 and k <= N // 2 + 2 and i == 0)
                # World-space position: x along +X per column k,
                # z decreases per row i (so the mesh dangles down).
                x = (k - N // 2) * GRID_PITCH
                z = z_top - i * GRID_PITCH
                mass = 0.0 if is_pinned else 1.0
                body = mb.add_body(
                    xform=wp.transform(
                        p=wp.vec3(x, 0.0, z), q=wp.quat_identity()
                    ),
                    mass=mass,
                )
                mb.add_shape_sphere(body, radius=SPHERE_RADIUS)
                self._bodies[k][i] = body
                self.register_body_extent(
                    body, (SPHERE_RADIUS, SPHERE_RADIUS, SPHERE_RADIUS)
                )

                # Connect to the body above (i - 1) and to the left
                # (k - 1) with revolute joints. The anchors are at the
                # shared midpoint so the joint rest length is zero.
                if i > 0:
                    upper = self._bodies[k][i - 1]
                    midpoint_z = z_top - (i - 0.5) * GRID_PITCH
                    self.add_joint(
                        body1=upper,
                        body2=body,
                        anchor1=(x, 0.0, midpoint_z),
                        # Axis = +Y (out-of-plane) so swinging happens
                        # in the XZ plane.
                        anchor2=(x, 1.0, midpoint_z),
                        mode=JointMode.REVOLUTE,
                    )
                if k > 0:
                    left = self._bodies[k - 1][i]
                    midpoint_x = (k - 0.5 - N // 2) * GRID_PITCH
                    self.add_joint(
                        body1=left,
                        body2=body,
                        anchor1=(midpoint_x, 0.0, z),
                        anchor2=(midpoint_x, 1.0, z),
                        mode=JointMode.REVOLUTE,
                    )

    def test_final(self) -> None:
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for row in self._bodies:
            for nbody in row:
                j = self._newton_to_jitter[nbody]
                assert np.isfinite(positions[j]).all(), f"body {nbody} non-finite"
                assert np.isfinite(velocities[j]).all(), f"body {nbody} non-finite"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
