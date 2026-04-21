# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter High Mass Ratio
#
# Port of Box2D's ``Robustness/HighMassRatio1`` sample. Three 10-base
# triangular pyramids of unit cubes, each topped by a single cube
# whose mass is 100, 200, or 300 times the rest. Canonical solver
# robustness test: low PGS iteration counts destabilise such stacks
# because the normal impulse on the lower rows has to overcome a huge
# friction budget at the cap.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_high_mass_ratio
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

BASE_COUNT = 10
BOX_HALF = 0.5
BOX_SPACING = 2.01 * BOX_HALF
# Mass of the single cube that crowns each pyramid.
CAP_MASSES = (100.0, 200.0, 300.0)
# Horizontal pitch between the three pyramid centres along +Y.
PYRAMID_PITCH = (BASE_COUNT + 2) * BOX_SPACING


class Example(DemoExample):
    def __init__(self, viewer, args):
        span = len(CAP_MASSES) * PYRAMID_PITCH
        cfg = DemoConfig(
            title="High Mass Ratio (100x, 200x, 300x caps)",
            camera_pos=(span * 0.8, span * 1.0, BASE_COUNT * BOX_SPACING * 1.4),
            camera_pitch=-22.0,
            camera_yaw=-55.0,
            fps=60,
            substeps=4,
            solver_iterations=24,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._bodies: list[int] = []

        y_origin = -0.5 * (len(CAP_MASSES) - 1) * PYRAMID_PITCH
        for idx, cap_mass in enumerate(CAP_MASSES):
            cy = y_origin + idx * PYRAMID_PITCH
            self._build_pyramid_with_cap(mb, cx=0.0, cy=cy, cap_mass=cap_mass)

    def _build_pyramid_with_cap(
        self, mb, cx: float, cy: float, cap_mass: float
    ) -> None:
        for level in range(BASE_COUNT):
            num_in_row = BASE_COUNT - level
            row_width = (num_in_row - 1) * BOX_SPACING
            for i in range(num_in_row):
                x = cx - row_width * 0.5 + i * BOX_SPACING
                z = level * BOX_SPACING + BOX_HALF + 1.0e-3
                body = mb.add_body(
                    xform=wp.transform(p=wp.vec3(x, cy, z), q=wp.quat_identity()),
                    mass=1.0,
                )
                mb.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
                self._bodies.append(body)
                self.register_body_extent(body, (BOX_HALF, BOX_HALF, BOX_HALF))

        # Heavy cap one cube above the tip of the pyramid.
        cap_z = BASE_COUNT * BOX_SPACING + BOX_HALF + 1.0e-3
        cap = mb.add_body(
            xform=wp.transform(p=wp.vec3(cx, cy, cap_z), q=wp.quat_identity()),
            mass=cap_mass,
        )
        mb.add_shape_box(cap, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
        self._bodies.append(cap)
        self.register_body_extent(cap, (BOX_HALF, BOX_HALF, BOX_HALF))

    def test_final(self) -> None:
        # No NaNs / explosions -- high mass ratios are the point of
        # this test so position tolerances are deliberately loose.
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody in self._bodies:
            j = self._newton_to_jitter[nbody]
            assert np.isfinite(positions[j]).all(), f"body {nbody} non-finite"
            assert np.isfinite(velocities[j]).all(), f"body {nbody} non-finite"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
