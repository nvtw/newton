# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX colorful mobile demo.
#
# Stylized version of the reference scene: a rigid gantry and table-like
# support hold two bright hanging chains plus a small block. The colored
# hanging pieces are actual rigid bodies tied by PhoenX ball-socket joint
# rows; the frame and platforms are static collision geometry.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_phoenx_mobile
###########################################################################

from __future__ import annotations

from itertools import pairwise

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    default_capsule_half_extents,
    run_ported_example,
)

# Static scene colors.
GRAY = (0.62, 0.65, 0.70)
GROUND = (0.78, 0.80, 0.84)
CYAN = (0.10, 0.78, 0.86)
YELLOW = (0.74, 0.67, 0.16)
PURPLE = (0.63, 0.18, 0.86)
GREEN = (0.24, 0.78, 0.38)
MAGENTA = (0.84, 0.18, 0.62)
RED = (0.76, 0.14, 0.18)
BLUE = (0.16, 0.34, 0.90)

LINK_COLORS = (
    (0.18, 0.80, 0.78),
    (0.97, 0.57, 0.13),
    (0.38, 0.18, 0.78),
    (0.20, 0.76, 0.30),
    (0.80, 0.18, 0.42),
    (0.13, 0.52, 0.88),
    (0.78, 0.86, 0.16),
    (0.94, 0.25, 0.76),
)

MAIN_LINKS = 10
MAIN_PITCH = 0.13
MAIN_RADIUS = 0.028
MAIN_HALF_HEIGHT = 0.045

SMALL_PITCH = 0.13
SMALL_RADIUS = 0.022
SMALL_HALF_HEIGHT = 0.040
SMALL_BLOCK_HE = 0.085


def _tf(x: float, y: float, z: float) -> wp.transform:
    return wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity())


def _world_box(
    builder: newton.ModelBuilder,
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    color: tuple[float, float, float],
    cfg: newton.ModelBuilder.ShapeConfig,
) -> None:
    builder.add_shape_box(
        -1,
        xform=_tf(*center),
        hx=half_extents[0],
        hy=half_extents[1],
        hz=half_extents[2],
        cfg=cfg,
        color=color,
    )


def _add_chain_link(
    builder: newton.ModelBuilder,
    center: tuple[float, float, float],
    radius: float,
    half_height: float,
    color: tuple[float, float, float],
) -> int:
    body = builder.add_link(xform=_tf(*center))
    builder.add_shape_capsule(body, radius=radius, half_height=half_height, color=color)
    return body


class Example(PortedExample):
    fps = 60
    sim_substeps = 16
    solver_iterations = 8
    velocity_iterations = 1
    gravity = (0.0, 0.0, -9.81)
    step_layout = "single_world"
    broad_phase = "explicit"
    default_friction = 0.65
    default_restitution = 0.0
    show_contacts = False
    start_paused = True

    def build_scene(self, builder: newton.ModelBuilder):
        static_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.7, restitution=0.0)
        builder.add_ground_plane(height=0.0, cfg=static_cfg, color=GROUND)

        # Table/support in the foreground.
        _world_box(builder, (0.10, 0.00, 0.05), (0.72, 0.48, 0.05), GRAY, static_cfg)
        _world_box(builder, (0.55, 0.00, 0.56), (0.09, 0.10, 0.46), GRAY, static_cfg)
        _world_box(builder, (-0.10, 0.00, 1.02), (0.66, 0.10, 0.07), GRAY, static_cfg)

        # Gantry and top anchor, matching the color-block style in the screenshot.
        _world_box(builder, (-1.47, 0.00, 0.09), (0.38, 0.16, 0.07), CYAN, static_cfg)
        _world_box(builder, (-1.58, 0.00, 0.80), (0.065, 0.065, 0.65), YELLOW, static_cfg)
        _world_box(builder, (-0.32, 0.00, 1.50), (1.20, 0.075, 0.065), PURPLE, static_cfg)
        _world_box(builder, (0.98, 0.00, 1.45), (0.27, 0.17, 0.12), GREEN, static_cfg)

        extents: list[tuple[float, float, float] | None] = []

        chain_a = self._add_vertical_chain(builder, anchor=(0.94, -0.17, 1.33), color_offset=0)
        chain_b = self._add_vertical_chain(builder, anchor=(1.24, 0.17, 1.33), color_offset=3)
        extents.extend(chain_a)
        extents.extend(chain_b)

        small = self._add_small_hanger(builder)
        extents.extend(small)

        return extents

    def _add_vertical_chain(
        self,
        builder: newton.ModelBuilder,
        *,
        anchor: tuple[float, float, float],
        color_offset: int,
    ) -> list[tuple[float, float, float]]:
        bodies: list[int] = []
        extents: list[tuple[float, float, float]] = []
        for i in range(MAIN_LINKS):
            z = anchor[2] - (i + 0.5) * MAIN_PITCH
            body = _add_chain_link(
                builder,
                (anchor[0], anchor[1], z),
                MAIN_RADIUS,
                MAIN_HALF_HEIGHT,
                LINK_COLORS[(i + color_offset) % len(LINK_COLORS)],
            )
            bodies.append(body)
            extents.append(default_capsule_half_extents(MAIN_RADIUS, MAIN_HALF_HEIGHT))

        joints: list[int] = []
        joints.append(
            builder.add_joint_ball(
                parent=-1,
                child=bodies[0],
                parent_xform=_tf(*anchor),
                child_xform=_tf(0.0, 0.0, MAIN_PITCH * 0.5),
                collision_filter_parent=False,
            )
        )
        for parent, child in pairwise(bodies):
            joints.append(
                builder.add_joint_ball(
                    parent=parent,
                    child=child,
                    parent_xform=_tf(0.0, 0.0, -MAIN_PITCH * 0.5),
                    child_xform=_tf(0.0, 0.0, MAIN_PITCH * 0.5),
                    collision_filter_parent=True,
                )
            )
        builder.add_articulation(joints)
        return extents

    def _add_small_hanger(self, builder: newton.ModelBuilder) -> list[tuple[float, float, float]]:
        anchor = (-0.86, -0.03, 0.84)
        link_a = _add_chain_link(
            builder,
            (anchor[0], anchor[1], anchor[2] - 0.5 * SMALL_PITCH),
            SMALL_RADIUS,
            SMALL_HALF_HEIGHT,
            RED,
        )
        link_b = _add_chain_link(
            builder,
            (anchor[0], anchor[1], anchor[2] - 1.5 * SMALL_PITCH),
            SMALL_RADIUS,
            SMALL_HALF_HEIGHT,
            BLUE,
        )

        block_center = (anchor[0], anchor[1], anchor[2] - 2.45 * SMALL_PITCH)
        block = builder.add_link(xform=_tf(*block_center))
        builder.add_shape_box(block, hx=SMALL_BLOCK_HE, hy=SMALL_BLOCK_HE, hz=SMALL_BLOCK_HE, color=MAGENTA)

        joints = [
            builder.add_joint_ball(
                parent=-1,
                child=link_a,
                parent_xform=_tf(*anchor),
                child_xform=_tf(0.0, 0.0, SMALL_PITCH * 0.5),
                collision_filter_parent=False,
            ),
            builder.add_joint_ball(
                parent=link_a,
                child=link_b,
                parent_xform=_tf(0.0, 0.0, -SMALL_PITCH * 0.5),
                child_xform=_tf(0.0, 0.0, SMALL_PITCH * 0.5),
                collision_filter_parent=True,
            ),
            builder.add_joint_ball(
                parent=link_b,
                child=block,
                parent_xform=_tf(0.0, 0.0, -SMALL_PITCH * 0.5),
                child_xform=_tf(0.0, 0.0, SMALL_BLOCK_HE),
                collision_filter_parent=True,
            ),
        ]
        builder.add_articulation(joints)

        return [
            default_capsule_half_extents(SMALL_RADIUS, SMALL_HALF_HEIGHT),
            default_capsule_half_extents(SMALL_RADIUS, SMALL_HALF_HEIGHT),
            default_box_half_extents(SMALL_BLOCK_HE, SMALL_BLOCK_HE, SMALL_BLOCK_HE),
        ]

    def test_final(self) -> None:
        positions = self.bodies.position.numpy()
        if not np.isfinite(positions).all():
            raise RuntimeError("mobile produced non-finite body positions")

        min_z = float(positions[1:, 2].min())
        if min_z < -0.25:
            raise RuntimeError(f"mobile body escaped below the scene: min_z={min_z:.3f}")

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(3.0, -3.6, 1.9), pitch=-18.0, yaw=138.0)


if __name__ == "__main__":
    run_ported_example(Example)
