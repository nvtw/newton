# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX tensegrity sculpture.
#
# Port of the PEEL TensegritySculpture test. A welded floating frame hangs
# from three ball-jointed capsule chains. The outer chains close loops back
# to the world, making this a compact full-coordinate joint stress test.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_tensegrity
###########################################################################

from __future__ import annotations

import math
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

CHAIN_RADIUS = 0.045
CHAIN_DENSITY = 2.0
FRAME_COLOR = (0.88, 0.32, 0.12)
CHAIN_COLOR = (0.18, 0.54, 0.78)
SUPPORT_COLOR = (0.48, 0.51, 0.56)


def _quat_z_to_dir(direction: np.ndarray) -> wp.quat:
    direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    sin_angle = float(np.linalg.norm(axis))
    cos_angle = float(np.dot(z_axis, direction))
    if sin_angle < 1.0e-8:
        if cos_angle > 0.0:
            return wp.quat_identity()
        return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
    axis /= sin_angle
    return wp.quat_from_axis_angle(wp.vec3(*axis), math.atan2(sin_angle, cos_angle))


def _xf(position) -> wp.transform:
    return wp.transform(
        p=wp.vec3(float(position[0]), float(position[1]), float(position[2])),
        q=wp.quat_identity(),
    )


class Example(PortedExample):
    sim_substeps = 40
    solver_iterations = 10
    velocity_iterations = 2
    step_layout = "single_world"
    broad_phase = "explicit"
    default_friction = 0.7
    show_contacts = False
    evaluate_fk = False

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane(color=(0.74, 0.76, 0.79))
        self._extents: list[tuple[float, float, float] | None] = []

        self._add_box(builder, -1, (3.00, 3.00, 0.22), (0.00, 0.0, 0.11), color=SUPPORT_COLOR)
        self._add_box(builder, -1, (0.34, 0.34, 2.70), (0.75, 0.0, 1.57), color=SUPPORT_COLOR)
        self._add_box(builder, -1, (1.80, 0.34, 0.30), (0.00, 0.0, 3.05), color=SUPPORT_COLOR)
        pos_base = np.array([0.0, 0.0, 0.11])
        pos_static_arm = np.array([0.0, 0.0, 3.05])

        pos_hook = np.array([-0.75, 0.0, 2.30])
        pos_arm = pos_hook + np.array([-0.74, 0.0, 0.0])
        pos_post = np.array([-1.62, 0.0, 3.57])
        pos_bar = np.array([-0.20, 0.0, 4.84])
        pos_crossbar = np.array([1.30, 0.0, 4.84])

        hook = self._add_box(builder, None, (0.32, 0.32, 0.32), pos_hook, 15.0, FRAME_COLOR)
        arm = self._add_box(builder, None, (1.20, 0.34, 0.34), pos_arm, 15.0, FRAME_COLOR)
        post = self._add_box(builder, None, (0.30, 0.30, 2.30), pos_post, 0.12, FRAME_COLOR)
        bar = self._add_box(builder, None, (3.10, 0.30, 0.28), pos_bar, 0.12, FRAME_COLOR)
        crossbar = self._add_box(builder, None, (0.30, 1.90, 0.26), pos_crossbar, 0.12, FRAME_COLOR)
        self.frame_bodies = [hook, arm, post, bar, crossbar]

        center_a = pos_static_arm + np.array([-0.82, 0.0, -0.15])
        center_b = pos_hook + np.array([0.0, 0.0, 0.16])
        center_links, center_lengths = self._add_chain_links(
            builder, center_a, center_b, self._chain_link_count(center_a, center_b)
        )

        outer_chains = []
        for y_sign in (-1.0, 1.0):
            outer_a = pos_crossbar + np.array([0.0, y_sign * 0.85, 0.0])
            outer_b = pos_base + np.array([1.25, y_sign * 1.25, 0.11])
            links, lengths = self._add_chain_links(builder, outer_a, outer_b, self._chain_link_count(outer_a, outer_b))
            outer_chains.append((links, lengths, outer_b, outer_a - pos_crossbar))

        tree = [
            builder.add_joint_ball(
                parent=-1,
                child=center_links[0],
                parent_xform=_xf(center_a),
                child_xform=_xf((0.0, 0.0, -0.5 * center_lengths[0])),
            )
        ]
        for index in range(len(center_links) - 1):
            tree.append(
                builder.add_joint_ball(
                    parent=center_links[index],
                    child=center_links[index + 1],
                    parent_xform=_xf((0.0, 0.0, 0.5 * center_lengths[index])),
                    child_xform=_xf((0.0, 0.0, -0.5 * center_lengths[index + 1])),
                )
            )
        tree.append(
            builder.add_joint_ball(
                parent=center_links[-1],
                child=hook,
                parent_xform=_xf((0.0, 0.0, 0.5 * center_lengths[-1])),
                child_xform=_xf(center_b - pos_hook),
            )
        )

        welds = (
            (hook, pos_hook, arm, pos_arm, pos_hook + np.array([-0.18, 0.0, 0.0])),
            (arm, pos_arm, post, pos_post, np.array([-1.62, 0.0, 2.30])),
            (post, pos_post, bar, pos_bar, np.array([-1.62, 0.0, 4.78])),
            (bar, pos_bar, crossbar, pos_crossbar, np.array([1.30, 0.0, 4.84])),
        )
        for parent, parent_pos, child, child_pos, anchor in welds:
            tree.append(
                builder.add_joint_fixed(
                    parent=parent,
                    child=child,
                    parent_xform=_xf(anchor - parent_pos),
                    child_xform=_xf(anchor - child_pos),
                )
            )

        loop_closers = []
        for links, lengths, outer_b, crossbar_anchor in outer_chains:
            tree.append(
                builder.add_joint_ball(
                    parent=crossbar,
                    child=links[0],
                    parent_xform=_xf(crossbar_anchor),
                    child_xform=_xf((0.0, 0.0, -0.5 * lengths[0])),
                )
            )
            for index in range(len(links) - 1):
                tree.append(
                    builder.add_joint_ball(
                        parent=links[index],
                        child=links[index + 1],
                        parent_xform=_xf((0.0, 0.0, 0.5 * lengths[index])),
                        child_xform=_xf((0.0, 0.0, -0.5 * lengths[index + 1])),
                    )
                )
            loop_closers.append((links[-1], lengths[-1], outer_b))

        builder.add_articulation(tree)
        for last_link, last_length, world_anchor in loop_closers:
            builder.add_joint_ball(
                parent=-1,
                child=last_link,
                parent_xform=_xf(world_anchor),
                child_xform=_xf((0.0, 0.0, 0.5 * last_length)),
            )

        return self._extents

    def _shape_cfg(self, density: float) -> newton.ModelBuilder.ShapeConfig:
        return newton.ModelBuilder.ShapeConfig(density=density, has_shape_collision=False)

    def _record_extent(self, body: int, extent: tuple[float, float, float]) -> None:
        while len(self._extents) <= body:
            self._extents.append(None)
        self._extents[body] = extent

    def _add_box(
        self,
        builder: newton.ModelBuilder,
        body: int | None,
        full_extents,
        position,
        density: float = 0.0,
        color=SUPPORT_COLOR,
    ) -> int:
        if body is None:
            body = builder.add_link(xform=_xf(position))
        xform = _xf(position) if body == -1 else None
        half_extents = tuple(0.5 * float(value) for value in full_extents)
        builder.add_shape_box(
            body,
            xform=xform,
            hx=half_extents[0],
            hy=half_extents[1],
            hz=half_extents[2],
            cfg=self._shape_cfg(density),
            color=color,
        )
        if body >= 0:
            self._record_extent(body, default_box_half_extents(*half_extents))
        return body

    def _chain_link_count(self, start, end) -> int:
        count = int(round(np.linalg.norm(np.asarray(end) - np.asarray(start)) / 0.30))
        return max(2, min(120, count))

    def _add_chain_links(self, builder: newton.ModelBuilder, start, end, count: int):
        points = [start + (end - start) * (index / count) for index in range(count + 1)]
        links: list[int] = []
        lengths: list[float] = []
        for point_a, point_b in pairwise(points):
            direction = point_b - point_a
            length = float(np.linalg.norm(direction))
            midpoint = 0.5 * (point_a + point_b)
            body = builder.add_link(
                xform=wp.transform(p=wp.vec3(*midpoint), q=_quat_z_to_dir(direction)),
            )
            half_height = 0.4 * length
            builder.add_shape_capsule(
                body,
                radius=CHAIN_RADIUS,
                half_height=half_height,
                cfg=self._shape_cfg(CHAIN_DENSITY),
                color=CHAIN_COLOR,
            )
            self._record_extent(body, default_capsule_half_extents(CHAIN_RADIUS, half_height))
            links.append(body)
            lengths.append(length)
        return links, lengths

    def configure_camera(self, viewer) -> None:
        viewer.set_camera(pos=wp.vec3(8.0, -8.0, 4.2), pitch=-8.0, yaw=135.0)

    def test_final(self) -> None:
        super().test_final()
        positions = self.bodies.position.numpy()
        frame_z = positions[np.asarray(self.frame_bodies) + 1, 2]
        if not ((frame_z > 1.0).all() and (frame_z < 6.0).all()):
            raise AssertionError(f"tensegrity frame left its hanging envelope: z={frame_z}")


if __name__ == "__main__":
    run_ported_example(Example)
