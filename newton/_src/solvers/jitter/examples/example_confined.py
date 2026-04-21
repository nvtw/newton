# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Confined
#
# Port of Box2D's ``Stacking/Confined`` sample. A tall closed room
# built from four static walls + floor + ceiling packed full of
# spheres. Stresses solver convergence with high contact density
# inside a closed box: every ball touches several neighbours and
# the walls, so the graph colouring has to find many small partitions.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_confined
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

# Half-extents of the confined box (inside dimensions).
BOX_HX = 3.0
BOX_HY = 3.0
BOX_HZ = 6.0
WALL_THICK = 0.2

SPHERE_RADIUS = 0.25
# Grid of spheres we attempt to drop into the box -- the solver
# has to pack them.
GRID_X = 8
GRID_Y = 8
GRID_Z = 16


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Confined",
            camera_pos=(BOX_HX * 4.0, BOX_HY * 4.0, BOX_HZ * 1.2),
            camera_pitch=-18.0,
            camera_yaw=-45.0,
            fps=60,
            substeps=4,
            solver_iterations=16,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        # ---- Static container (one body, six plank shapes) ----
        self._container = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, BOX_HZ), q=wp.quat_identity()
            ),
            mass=0.0,
        )
        # +X wall
        mb.add_shape_box(
            self._container,
            hx=WALL_THICK,
            hy=BOX_HY + WALL_THICK,
            hz=BOX_HZ,
            xform=wp.transform(
                p=wp.vec3(+BOX_HX, 0.0, 0.0), q=wp.quat_identity()
            ),
        )
        # -X wall
        mb.add_shape_box(
            self._container,
            hx=WALL_THICK,
            hy=BOX_HY + WALL_THICK,
            hz=BOX_HZ,
            xform=wp.transform(
                p=wp.vec3(-BOX_HX, 0.0, 0.0), q=wp.quat_identity()
            ),
        )
        # +Y wall
        mb.add_shape_box(
            self._container,
            hx=BOX_HX + WALL_THICK,
            hy=WALL_THICK,
            hz=BOX_HZ,
            xform=wp.transform(
                p=wp.vec3(0.0, +BOX_HY, 0.0), q=wp.quat_identity()
            ),
        )
        # -Y wall
        mb.add_shape_box(
            self._container,
            hx=BOX_HX + WALL_THICK,
            hy=WALL_THICK,
            hz=BOX_HZ,
            xform=wp.transform(
                p=wp.vec3(0.0, -BOX_HY, 0.0), q=wp.quat_identity()
            ),
        )
        # Floor (inside the room)
        mb.add_shape_box(
            self._container,
            hx=BOX_HX,
            hy=BOX_HY,
            hz=WALL_THICK,
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, -BOX_HZ), q=wp.quat_identity()
            ),
        )

        # ---- Spheres ----
        self._spheres: list[int] = []
        d = 2.0 * SPHERE_RADIUS + 0.02
        x_origin = -(GRID_X - 1) * 0.5 * d
        y_origin = -(GRID_Y - 1) * 0.5 * d
        z_origin = BOX_HZ - 0.5 + 0.5  # start near top of the room
        for k in range(GRID_Z):
            for i in range(GRID_X):
                for j in range(GRID_Y):
                    pos = wp.vec3(
                        x_origin + i * d,
                        y_origin + j * d,
                        z_origin + k * d,
                    )
                    body = mb.add_body(
                        xform=wp.transform(p=pos, q=wp.quat_identity()),
                        mass=0.1,
                    )
                    mb.add_shape_sphere(body, radius=SPHERE_RADIUS)
                    self._spheres.append(body)
                    self.register_body_extent(
                        body, (SPHERE_RADIUS, SPHERE_RADIUS, SPHERE_RADIUS)
                    )

    def test_final(self) -> None:
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody in self._spheres:
            j = self._newton_to_jitter[nbody]
            assert np.isfinite(positions[j]).all(), f"body {nbody} non-finite"
            assert np.isfinite(velocities[j]).all(), f"body {nbody} non-finite"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
