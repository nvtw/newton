# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX card house.
#
# A five-storey house assembled from freely moving, full-coordinate cards.
# Every level contains leaning pairs and horizontal cards that support the
# next level. The small thickness and high friction make this a useful
# contact-stability scene without relying on joints or planar constraints.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_card_house
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

CARD_HALF_HEIGHT = 0.50
CARD_HALF_WIDTH = 0.34
CARD_HALF_THICKNESS = 0.018
LEAN_ANGLE = math.radians(13.0)
STORIES = 5

CARD_COLORS = (
    (0.82, 0.18, 0.20),
    (0.94, 0.68, 0.16),
    (0.16, 0.50, 0.78),
    (0.20, 0.66, 0.42),
)


def _quat_y(angle: float) -> wp.quat:
    half_angle = 0.5 * angle
    return wp.quat(0.0, math.sin(half_angle), 0.0, math.cos(half_angle))


class Example(PortedExample):
    sim_substeps = 20
    solver_iterations = 10
    velocity_iterations = 2
    step_layout = "single_world"
    broad_phase = "sap"
    shape_pairs_max = 4096
    default_friction = 0.82
    show_contacts = False

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane(color=(0.72, 0.74, 0.77))

        extents: list[tuple[float, float, float]] = []
        center_offset = CARD_HALF_HEIGHT * math.sin(LEAN_ANGLE) + CARD_HALF_THICKNESS
        pair_spacing = 2.0 * center_offset + 0.025
        storey_height = 2.0 * CARD_HALF_HEIGHT * math.cos(LEAN_ANGLE) + 2.0 * CARD_HALF_THICKNESS

        for storey in range(STORIES):
            pair_count = STORIES - storey
            base_z = storey * storey_height
            x_start = -0.5 * (pair_count - 1) * 2.0 * pair_spacing

            for pair_index in range(pair_count):
                pair_x = x_start + pair_index * 2.0 * pair_spacing
                for sign in (-1.0, 1.0):
                    center = wp.vec3(
                        pair_x + sign * center_offset,
                        0.0,
                        base_z + CARD_HALF_HEIGHT * math.cos(LEAN_ANGLE) + CARD_HALF_THICKNESS,
                    )
                    body = builder.add_body(
                        xform=wp.transform(p=center, q=_quat_y(-sign * LEAN_ANGLE)),
                    )
                    builder.add_shape_box(
                        body,
                        hx=CARD_HALF_THICKNESS,
                        hy=CARD_HALF_WIDTH,
                        hz=CARD_HALF_HEIGHT,
                        color=CARD_COLORS[(storey + pair_index) % len(CARD_COLORS)],
                    )
                    extents.append(default_box_half_extents(CARD_HALF_THICKNESS, CARD_HALF_WIDTH, CARD_HALF_HEIGHT))

                if pair_index + 1 < pair_count:
                    half_span = pair_spacing
                    body = builder.add_body(
                        xform=wp.transform(
                            p=wp.vec3(pair_x + pair_spacing, 0.0, base_z + storey_height),
                            q=wp.quat_identity(),
                        ),
                    )
                    builder.add_shape_box(
                        body,
                        hx=half_span,
                        hy=CARD_HALF_WIDTH,
                        hz=CARD_HALF_THICKNESS,
                        color=CARD_COLORS[(storey + pair_index + 2) % len(CARD_COLORS)],
                    )
                    extents.append(default_box_half_extents(half_span, CARD_HALF_WIDTH, CARD_HALF_THICKNESS))

        self._card_count = len(extents)
        return extents

    def configure_camera(self, viewer) -> None:
        viewer.set_camera(pos=wp.vec3(7.2, -7.2, 3.7), pitch=-12.0, yaw=135.0)

    def test_final(self) -> None:
        super().test_final()
        positions = self.bodies.position.numpy()[1 : self._card_count + 1]
        min_z = float(positions[:, 2].min())
        max_radius = float(np.linalg.norm(positions[:, :2], axis=1).max())
        if min_z < -0.25:
            raise AssertionError(f"card escaped below the ground: min_z={min_z:.3f}")
        if max_radius > 8.0:
            raise AssertionError(f"card escaped the scene: max_radius={max_radius:.3f}")


if __name__ == "__main__":
    run_ported_example(Example)
