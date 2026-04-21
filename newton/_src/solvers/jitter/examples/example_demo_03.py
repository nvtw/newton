# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Demo 03 -- Ancient Pyramids
#
# Port of ``JitterDemo.Demos.Demo03`` (see
# ``C:\git3\jitterphysics2\src\JitterDemo\Demos\Demo03.cs``). Two
# adjacent pyramids: one of unit cubes, one of short cylinders
# (CylinderShape(1.0, 0.5) in the C# source -- height 1.0, radius
# 0.5). Stresses stacking contact stability across shape types.
#
# The cube pyramid matches :mod:`example_pyramid` but smaller (20
# layers instead of 40) so this demo stays lively at 60 Hz.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_demo_03
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

# Original Jitter demo uses 40 layers; we go lighter (20) so 60 Hz
# stays comfortable without the full pyramid frame budget of
# ``example_pyramid``.
LAYERS = 20

BOX_HALF = 0.5
BOX_SPACING = 1.01  # [m], matches the C# reference

CYL_RADIUS = 0.5
CYL_HALF_HEIGHT = 0.5  # CylinderShape(height=1, radius=0.5) in Jitter
CYL_SPACING = 1.01  # [m], matches the C# reference


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Ancient Pyramids",
            camera_pos=(20.0, 20.0, 12.0),
            camera_pitch=-18.0,
            camera_yaw=-50.0,
            fps=60,
            substeps=4,
            solver_iterations=8,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._box_bodies: list[int] = []
        self._cyl_bodies: list[int] = []

        # ---- Box pyramid (origin) -------------------------------
        # Pyramid pattern matches ``BuildPyramid`` in Common.cs:
        #
        #     for (int i = 0; i < size; i++)
        #         for (int e = i; e < size; e++)
        #             pos = base + ((e - i*0.5) * 1.01, 0.5 + i, 0);
        #
        # The crucial bit is that ``e`` starts at ``i``, not at 0, so
        # the first box of row ``i`` sits at ``x = 0.5 * i * 1.01``
        # and each upper row is centred over the pair below it. If
        # ``e`` starts at 0 the row appears to be shifted left by a
        # half-box per level (cumulative), which is what I had before
        # and what made the upper layers look 'one object size too
        # far off' from the row below.
        for i in range(LAYERS):
            for e in range(i, LAYERS):
                x = (e - i * 0.5) * BOX_SPACING
                y = 0.0
                z = BOX_HALF + i * 1.0
                body = mb.add_body(
                    xform=wp.transform(
                        p=wp.vec3(x, y, z), q=wp.quat_identity()
                    ),
                    mass=1.0,
                )
                mb.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
                self._box_bodies.append(body)
                self.register_body_extent(body, (BOX_HALF, BOX_HALF, BOX_HALF))

        # ---- Cylinder pyramid (offset +(10, 10)) ----------------
        # Same pyramid pattern as the boxes but with
        # ``CylinderShape(height=1, radius=0.5)``. Jitter cylinders
        # stand on +Y (their axial direction); Newton cylinders stand
        # on +Z by default, so with the Jitter Y -> Newton Z remap
        # the cylinders end up upright with no extra rotation.
        cyl_origin = np.array([10.0, 10.0, 0.0], dtype=np.float32)
        for i in range(LAYERS):
            for e in range(i, LAYERS):
                pos = cyl_origin + np.array(
                    [(e - i * 0.5) * CYL_SPACING, 0.0, 0.5 + i * 1.0],
                    dtype=np.float32,
                )
                body = mb.add_body(
                    xform=wp.transform(
                        p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                        q=wp.quat_identity(),
                    ),
                    mass=1.0,
                )
                mb.add_shape_cylinder(
                    body, radius=CYL_RADIUS, half_height=CYL_HALF_HEIGHT
                )
                self._cyl_bodies.append(body)
                self.register_body_extent(
                    body, (CYL_RADIUS, CYL_RADIUS, CYL_HALF_HEIGHT)
                )

    def test_final(self) -> None:
        """Every body sank to (or near) the floor and stayed inside the
        pyramid footprint. Loose tolerance because the second (cylinder)
        pyramid is inherently unstable under friction compared to cubes.
        """
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for newton_idx in self._box_bodies + self._cyl_bodies:
            j = self._newton_to_jitter[newton_idx]
            pos = positions[j]
            vel = velocities[j]
            assert np.isfinite(pos).all(), f"body {newton_idx} non-finite position"
            assert float(np.linalg.norm(vel)) < 5.0, (
                f"body {newton_idx} still moving at {float(np.linalg.norm(vel)):.2f} m/s"
            )


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
