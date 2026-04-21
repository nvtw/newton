# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Circle Stack
#
# Port of Box2D's ``Stacking/Circle Stack`` sample: a short vertical
# column of spheres, each heavier than the one below. Tests the
# contact solver against a small but high-mass-ratio stack with
# point contacts rather than face contacts. Sphere restitution is
# non-zero in Box2D; Jitter doesn't implement restitution yet, so
# we drop that aspect and keep the mass ramp so the heavy top
# sphere presses down through its lighter neighbours.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_circle_stack
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

SPHERE_COUNT = 4
SPHERE_RADIUS = 0.5
# Vertical pitch slightly over 2r so the settling pass closes the
# 0.25 m gap and builds a warm-started contact per pair.
START_PITCH = 1.25


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Circle Stack",
            camera_pos=(4.0, 6.0, 4.0),
            camera_pitch=-18.0,
            camera_yaw=-60.0,
            fps=60,
            substeps=4,
            solver_iterations=16,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._sphere_bodies: list[int] = []
        z = SPHERE_RADIUS + 0.05
        for i in range(SPHERE_COUNT):
            mass = 1.0 + 4.0 * i  # 1, 5, 9, 13 -- matches Box2D sample
            body = mb.add_body(
                xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()),
                mass=mass,
            )
            mb.add_shape_sphere(body, radius=SPHERE_RADIUS)
            self._sphere_bodies.append(body)
            self.register_body_extent(
                body, (SPHERE_RADIUS, SPHERE_RADIUS, SPHERE_RADIUS)
            )
            z += START_PITCH

    def test_final(self) -> None:
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody in self._sphere_bodies:
            j = self._newton_to_jitter[nbody]
            assert np.isfinite(positions[j]).all(), f"body {nbody} non-finite pos"
            assert np.isfinite(velocities[j]).all(), f"body {nbody} non-finite vel"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
