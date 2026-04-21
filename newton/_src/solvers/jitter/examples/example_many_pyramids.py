# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Many Pyramids
#
# Port of Box2D's ``Benchmark/Many Pyramids`` scene
# (``shared/benchmarks.c::CreateManyPyramids``). Arranges ``rows *
# columns`` small box pyramids side-by-side on a ground plane. Every
# pyramid is identical -- a 10-layer (1 m box) triangular stack --
# which is a good parallel stress test for the graph colouring and
# the single-block iterate kernel: ``rows * columns`` independent
# sub-islands multiply the per-colour constraint count without
# changing the pyramid-specific convergence properties.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_many_pyramids
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

# Each pyramid is a 10-base triangular stack of unit cubes.
BASE_COUNT = 10
# Grid of pyramids. 3 x 3 = 9 independent pyramids keeps the scene
# responsive on a laptop GPU; crank up ``ROW_COUNT`` / ``COLUMN_COUNT``
# for a harder benchmark.
ROW_COUNT = 3
COLUMN_COUNT = 3

BOX_HALF = 0.5
BOX_SPACING = 2.01 * BOX_HALF
# Horizontal pitch between pyramid centres. ``(base + 2)`` leaves a
# 2-cube-wide gap between adjacent pyramid bases so their contact
# islands are genuinely independent.
PYRAMID_PITCH = (BASE_COUNT + 2) * BOX_SPACING


class Example(DemoExample):
    def __init__(self, viewer, args):
        span = max(ROW_COUNT, COLUMN_COUNT) * PYRAMID_PITCH
        cfg = DemoConfig(
            title="Many Pyramids",
            camera_pos=(span * 1.1, span * 1.1, BASE_COUNT * BOX_SPACING * 1.5),
            camera_pitch=-25.0,
            camera_yaw=-45.0,
            fps=60,
            substeps=4,
            solver_iterations=20,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._box_bodies: list[int] = []
        self._nominal_positions: list[tuple[float, float, float]] = []

        # Centre the whole grid around the world origin.
        x_origin = -0.5 * (COLUMN_COUNT - 1) * PYRAMID_PITCH
        y_origin = -0.5 * (ROW_COUNT - 1) * PYRAMID_PITCH

        for row in range(ROW_COUNT):
            for col in range(COLUMN_COUNT):
                cx = x_origin + col * PYRAMID_PITCH
                cy = y_origin + row * PYRAMID_PITCH
                self._build_pyramid(mb, cx, cy)

    def _build_pyramid(self, mb, cx: float, cy: float) -> None:
        """One triangular pyramid centred at ``(cx, cy)`` on the ground."""
        for level in range(BASE_COUNT):
            num_in_row = BASE_COUNT - level
            row_width = (num_in_row - 1) * BOX_SPACING
            for i in range(num_in_row):
                x = cx - row_width * 0.5 + i * BOX_SPACING
                y = cy
                # Start each cube a hair above its resting height.
                z = level * BOX_SPACING + BOX_HALF + 1.0e-3
                body = mb.add_body(
                    xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()),
                    mass=1.0,
                )
                mb.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
                self._box_bodies.append(body)
                self._nominal_positions.append(
                    (x, y, level * BOX_SPACING + BOX_HALF)
                )
                self.register_body_extent(body, (BOX_HALF, BOX_HALF, BOX_HALF))

    def test_final(self) -> None:
        pos_tol = 0.15
        vel_tol = 0.5
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody, nominal in zip(self._box_bodies, self._nominal_positions, strict=True):
            jbody = self._newton_to_jitter[nbody]
            pos = positions[jbody]
            vel = velocities[jbody]
            assert np.isfinite(pos).all(), f"body {nbody} position non-finite ({pos})"
            disp = float(np.linalg.norm(pos - np.asarray(nominal, dtype=np.float32)))
            speed = float(np.linalg.norm(vel))
            assert disp < pos_tol, f"body {nbody} displaced {disp:.3f} m from {nominal}"
            assert speed < vel_tol, f"body {nbody} still moving at {speed:.3f} m/s"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
