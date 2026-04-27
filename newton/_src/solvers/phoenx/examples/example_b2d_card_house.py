# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Stacking / Card House``
#
# 3-storey house of cards. Each "card" is a thin tall plank leaned against
# its partner; horizontal cards span pairs. The 2D demo's cards are thin
# rectangles; in 3D we use thin boxes oriented around +y.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_card_house
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

# Card geometry: thin (z very small), tall (x = card_h), narrow in y.
CARD_H = 0.5
CARD_T = 0.04
CARD_W = 0.5
LEAN_DEG = 12.0
N_STORIES = 3


def _quat_y(angle_rad: float):
    half = 0.5 * angle_rad
    return wp.quat(0.0, math.sin(half), 0.0, math.cos(half))


class Example(PortedExample):
    sim_substeps = 12
    solver_iterations = 24
    default_friction = 0.7

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        lean = math.radians(LEAN_DEG)

        spacing = 2.0 * CARD_H * math.sin(lean) + 0.04
        story_h = 2.0 * CARD_H * math.cos(lean) + 2.0 * CARD_T

        for story in range(N_STORIES):
            n_pairs = N_STORIES - story
            base_z = story * story_h
            x0 = -((n_pairs - 1) * 2.0 * spacing) * 0.5
            for p in range(n_pairs):
                xc = x0 + p * 2.0 * spacing
                # Two leaning cards forming a tent.
                for sign, ang in ((1, +lean), (-1, -lean)):
                    cx = xc + sign * (CARD_H * math.sin(lean) + CARD_T)
                    cz = base_z + CARD_H * math.cos(lean) + CARD_T
                    body = builder.add_body(
                        xform=wp.transform(p=wp.vec3(cx, 0.0, cz), q=_quat_y(ang)),
                    )
                    builder.add_shape_box(body, hx=CARD_T, hy=CARD_W, hz=CARD_H)
                    extents.append(default_box_half_extents(CARD_T, CARD_W, CARD_H))
                # Horizontal card on top of the pair.
                top_z = base_z + 2 * CARD_H * math.cos(lean) + CARD_T
                if p < n_pairs - 1:
                    bx = xc + spacing
                    bw = spacing  # reaches the next pair's centre
                    body = builder.add_body(
                        xform=wp.transform(p=wp.vec3(bx, 0.0, top_z), q=wp.quat_identity()),
                    )
                    builder.add_shape_box(body, hx=bw, hy=CARD_W, hz=CARD_T)
                    extents.append(default_box_half_extents(bw, CARD_W, CARD_T))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(6.0, 0.0, 2.5), pitch=-10.0, yaw=180.0)


if __name__ == "__main__":
    run_ported_example(Example)
