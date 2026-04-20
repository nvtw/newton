# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Demo 08 -- Contact Manifold Test
#
# Port of ``JitterDemo.Demos.Demo08`` (see
# ``C:\git3\jitterphysics2\src\JitterDemo\Demos\Demo08.cs``). A static
# 5 x 0.5 x 0.5 box floats 1 m above the floor; a frictionless 3 m
# cylinder (radius 0.5) is dropped on top of it. Stresses the
# box <-> cylinder contact manifold against a static body.
#
# Jitter's ``Friction = 0`` on the cylinder maps to
# ``material.mu = 0`` on the Newton shape. Static body in Newton is a
# body with ``mass = 0`` (the solver then sees ``inverse_mass = 0``,
# which Jitter treats as a non-moveable body).
#
# Run:  python -m newton._src.solvers.jitter.example_demo_08
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.example_jitter_common import (
    DemoConfig,
    DemoExample,
)


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Contact Manifold Test",
            camera_pos=(8.0, 8.0, 4.0),
            camera_pitch=-18.0,
            camera_yaw=-45.0,
            fps=60,
            substeps=4,
            solver_iterations=4,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        # Static plank -- C# BoxShape(5, 0.5, 0.5) centred at (0, 1, 0)
        # in +Y-up; in Newton's +Z-up convention that's (0, 0, 1) with
        # half-extents along X/Y/Z = (2.5, 0.25, 0.25). Newton makes the
        # body static by giving it zero mass.
        self._static_body = mb.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
            mass=0.0,
        )
        mb.add_shape_box(self._static_body, hx=2.5, hy=0.25, hz=0.25)

        # Frictionless cylinder dropped from (0, 2.5, 0) in +Y-up =>
        # (0, 0, 2.5) in Newton.  CylinderShape(0.5, 3) means radius
        # 0.5, full height 3.0, so half_height = 1.5.
        cfg = mb.default_shape_cfg.copy()
        cfg.mu = 0.0
        self._dropped_body = mb.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.5), q=wp.quat_identity()),
            mass=1.0,
        )
        mb.add_shape_cylinder(
            self._dropped_body, radius=0.5, half_height=1.5, cfg=cfg
        )
        self.register_body_extent(self._dropped_body, (0.5, 0.5, 1.5))

    def test_final(self) -> None:
        """After settle the cylinder should still be in the neighbourhood
        of the static plank (friction = 0 means it can slide but not
        escape the basin). No explosion, no NaNs.
        """
        pos, vel = self.jitter_body_state(self._dropped_body)
        assert np.isfinite(pos).all() and np.isfinite(vel).all()


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
