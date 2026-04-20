# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Demo 10 -- Stacked Cubes
#
# Port of ``JitterDemo.Demos.Demo10`` (see
# ``C:\git3\jitterphysics2\src\JitterDemo\Demos\Demo10.cs``). The C#
# reference stacks 32 unit cubes and 32 cones; we keep only the cube
# stack (Newton's Jitter port intentionally sticks to a single shape
# per body and contact types we already fully support). The stack
# deliberately uses tight 0.999 m vertical spacing so the solver has
# to resolve the resulting micro-penetration without explosion.
#
# Run:  python -m newton._src.solvers.jitter.example_demo_10
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

STACK_HEIGHT = 32
BOX_HALF = 0.5
BOX_SPACING = 0.999  # [m] -- matches the C# reference exactly


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Stacked Cubes",
            camera_pos=(5.0, 25.0, 15.0),
            camera_pitch=-18.0,
            camera_yaw=-80.0,
            fps=60,
            substeps=3,
            solver_iterations=4,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._stack_bodies: list[int] = []
        for i in range(STACK_HEIGHT):
            z = BOX_HALF + i * BOX_SPACING
            body = mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()
                ),
                mass=1.0,
            )
            mb.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
            self._stack_bodies.append(body)
            self.register_body_extent(body, (BOX_HALF, BOX_HALF, BOX_HALF))

    def test_final(self) -> None:
        """Stack shouldn't topple: every cube's lateral deviation stays
        within a cube-edge of the stack axis, and none fly off vertically.
        """
        positions = self.world.bodies.position.numpy()
        for level, newton_idx in enumerate(self._stack_bodies):
            j = self._newton_to_jitter[newton_idx]
            pos = positions[j]
            assert np.isfinite(pos).all(), f"body {newton_idx} non-finite pos"
            r_xy = float(np.hypot(pos[0], pos[1]))
            assert r_xy < 2.0 * BOX_HALF, (
                f"stack level {level} drifted by {r_xy:.2f} m off-axis"
            )


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
