# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Demo 02 -- Tower of Jitter
#
# Port of ``JitterDemo.Demos.Demo02`` (see
# ``C:\git3\jitterphysics2\src\JitterDemo\Demos\Demo02.cs``). A single
# tall tower of 40 rings, each ring laying 32 thin planks in a circle
# with a 1/64 * 2 PI half-rotation offset between rings. The tower
# starts stable and is a classic solver-stability stress test.
#
# Jitter uses +Y up; Newton uses +Z up, so the tower rises along +Z
# and the rings are constructed by rotating about +Z.
#
# Uses the shared :mod:`example_jitter_common` plumbing so the scene
# builder only has to populate the :class:`newton.ModelBuilder` and
# supply a camera.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_demo_02
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

TOWER_HEIGHT_LAYERS = 40
BOXES_PER_RING = 32
# Plank half-extents. The Jitter C# reference uses
# ``BoxShape(3, 1, 0.2)`` in its +Y-up convention, where:
#   * ``X = 3.0`` is tangential (the long edge that runs around the ring),
#   * ``Y = 1.0`` is the *vertical* brick height (this is what the
#     ``0.5 + e`` vertical stepping relies on),
#   * ``Z = 0.2`` is the thin radial wall thickness.
# Newton uses +Z-up, so Jitter's Y (height) must become Newton's Z,
# and Jitter's Z (radial thickness) becomes Newton's Y. Getting this
# swap wrong makes the bricks look like flat paving stones laid on
# their side and leaves vertical gaps of ~0.8 m between layers, so
# the tower instantly collapses.
PLANK_HX = 1.5
PLANK_HY = 0.1
PLANK_HZ = 0.5
# Ring radius from the C# source (the JVector(0, 0.5+e, 19.5) puts the
# ring centre at radius 19.5). Jitter uses raw half-extents in these
# offsets so nothing needs rescaling.
RING_RADIUS = 19.5
HALF_ROTATION_STEP = 2.0 * math.pi / 64.0
FULL_ROTATION_STEP = 2.0 * HALF_ROTATION_STEP


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Tower of Jitter",
            # Tower is 40 layers * 1 m tall with ring radius 19.5 m;
            # stand outside along +X, aim back toward origin (-X).
            camera_pos=(55.0, 0.0, 22.0),
            camera_pitch=-10.0,
            camera_yaw=180.0,
            fps=60,
            substeps=4,
            solver_iterations=12,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._plank_bodies: list[int] = []
        orientation_rad = 0.0
        for e in range(TOWER_HEIGHT_LAYERS):
            orientation_rad += HALF_ROTATION_STEP
            for i in range(BOXES_PER_RING):
                # Jitter code: position = pos + Transform((0, 0.5+e, 19.5),
                # orientation). In Newton's +Z-up frame, +Y becomes +Z
                # and the rotation axis is +Z.
                cos_o = math.cos(orientation_rad)
                sin_o = math.sin(orientation_rad)
                local = np.array([0.0, RING_RADIUS, 0.5 + e], dtype=np.float32)
                # Rotate local by orientation_rad about +Z.
                world = np.array(
                    [
                        cos_o * local[0] - sin_o * local[1],
                        sin_o * local[0] + cos_o * local[1],
                        local[2],
                    ],
                    dtype=np.float32,
                )
                quat = wp.quat_from_axis_angle(
                    wp.vec3(0.0, 0.0, 1.0), orientation_rad
                )
                body = mb.add_body(
                    xform=wp.transform(
                        p=wp.vec3(float(world[0]), float(world[1]), float(world[2])),
                        q=quat,
                    ),
                    mass=1.0,
                )
                mb.add_shape_box(body, hx=PLANK_HX, hy=PLANK_HY, hz=PLANK_HZ)
                self._plank_bodies.append(body)
                self.register_body_extent(body, (PLANK_HX, PLANK_HY, PLANK_HZ))
                orientation_rad += FULL_ROTATION_STEP

    def test_final(self) -> None:
        """After settle, every plank should still be inside a reasonable
        envelope around the original tower. A full-tower collapse would
        scatter bodies well beyond ``tolerance`` m, so this catches solver
        blow-ups without being strict about the exact stack geometry.
        """
        tolerance = RING_RADIUS * 3.0
        positions = self.world.bodies.position.numpy()
        for newton_idx in self._plank_bodies:
            j = self._newton_to_jitter[newton_idx]
            pos = positions[j]
            assert np.isfinite(pos).all(), f"body {newton_idx} non-finite position"
            r_xy = float(math.hypot(pos[0], pos[1]))
            assert r_xy < tolerance, (
                f"plank {newton_idx} flew outside the tower envelope "
                f"(r_xy={r_xy:.2f}, tol={tolerance:.2f})"
            )


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
