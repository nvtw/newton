# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX suspension bridge.
#
# Port of the PEEL SuspensionBridge test. Capsule-link main cables carry a
# ball-jointed plank deck through vertical hanger ropes. The multiply
# connected graph is solved directly in full coordinates, including cargo
# dropped onto the center of the span.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_suspension_bridge
###########################################################################

from __future__ import annotations

import math
from itertools import pairwise
from typing import ClassVar

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    default_capsule_half_extents,
    run_ported_example,
)

TOWER_COLOR = (0.48, 0.50, 0.54)
CABLE_COLOR = (0.15, 0.20, 0.24)
DECK_COLOR = (0.66, 0.38, 0.16)
CARGO_COLOR = (0.82, 0.23, 0.16)


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
    sim_substeps = 25
    solver_iterations = 10
    velocity_iterations = 2
    step_layout = "single_world"
    broad_phase = "sap"
    shape_pairs_max = 20000
    default_friction = 0.75
    show_contacts = False
    finalize_kwargs: ClassVar[dict[str, bool]] = {"skip_validation_joints": True}
    evaluate_fk = False

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane(color=(0.70, 0.72, 0.74))
        self._extents: list[tuple[float, float, float] | None] = []

        plank_count = 14
        cargo_count = 4
        length_scale = plank_count / 14.0
        height_scale = 1.0 if length_scale < 1.0 else math.sqrt(length_scale)
        plank_length, plank_width, plank_thickness = 1.3, 2.6, 0.14
        span = plank_count * plank_length
        self.deck_z = 2.6 * height_scale
        tower_x = 0.5 * span + 0.6
        tower_top_z = self.deck_z + 3.2 + 1.6 * length_scale
        anchor_x = tower_x + 3.2 + 1.0 * length_scale
        cable_y = 0.5 * plank_width + 0.12
        sag = (tower_top_z - self.deck_z) * 0.62

        for tower_position in (-tower_x, tower_x):
            for leg_y in (-cable_y, cable_y):
                self._add_box(
                    builder,
                    -1,
                    (0.45, 0.45, tower_top_z),
                    (tower_position, leg_y, 0.5 * tower_top_z),
                    color=TOWER_COLOR,
                )
            self._add_box(
                builder,
                -1,
                (0.45, 2.0 * cable_y + 0.45, 0.4),
                (tower_position, 0.0, tower_top_z - 1.1),
                color=TOWER_COLOR,
            )
            ground_anchor_x = anchor_x if tower_position > 0.0 else -anchor_x
            self._add_box(
                builder,
                -1,
                (1.2, 2.0 * cable_y + 0.6, 0.7),
                (ground_anchor_x, 0.0, 0.35),
                color=TOWER_COLOR,
            )

        sample_count = 2 * plank_count
        cable_links = [None, None]
        cable_midpoints = [None, None]
        cable_x_positions = [None, None]
        for side_index, y_sign in enumerate((-1.0, 1.0)):
            y = y_sign * cable_y
            for x_sign in (-1.0, 1.0):
                start = np.array([x_sign * anchor_x, y, 0.45])
                end = np.array([x_sign * tower_x, y, tower_top_z])
                side_points = [start + (end - start) * (index / 5.0) for index in range(6)]
                self._add_rope(builder, side_points, 0.07, 3.0, (-1, None), (-1, None))

            main_points = []
            for index in range(sample_count + 1):
                fraction = index / sample_count
                x = -tower_x + fraction * 2.0 * tower_x
                normalized_x = x / tower_x
                main_points.append(np.array([x, y, tower_top_z - sag * (1.0 - normalized_x * normalized_x)]))
            links, midpoints = self._add_rope(builder, main_points, 0.07, 3.0, (-1, None), (-1, None))
            cable_links[side_index] = links
            cable_midpoints[side_index] = midpoints
            cable_x_positions[side_index] = np.asarray([midpoint[0] for midpoint in midpoints])

        self.planks: list[int] = []
        hinge_y = (-0.42 * plank_width, 0.42 * plank_width)
        previous_plank = None
        for index in range(plank_count):
            x = -0.5 * span + (index + 0.5) * plank_length
            plank = self._add_box(
                builder,
                None,
                (0.97 * plank_length, plank_width, plank_thickness),
                (x, 0.0, self.deck_z),
                density=0.7,
                collision_group=-2,
                color=DECK_COLOR,
            )
            self.planks.append(plank)
            if previous_plank is not None:
                for y in hinge_y:
                    builder.add_joint_ball(
                        parent=previous_plank,
                        child=plank,
                        parent_xform=_xf((0.5 * plank_length, y, 0.0)),
                        child_xform=_xf((-0.5 * plank_length, y, 0.0)),
                    )
            previous_plank = plank

        for end_index, x_sign in ((0, -1.0), (plank_count - 1, 1.0)):
            end_plank = self.planks[end_index]
            ramp_position = np.array([x_sign * (0.5 * span + 0.85), 0.0, self.deck_z - 0.1])
            self._add_box(builder, -1, (1.6, plank_width, 0.3), ramp_position, color=DECK_COLOR)
            for y in hinge_y:
                builder.add_joint_ball(
                    parent=-1,
                    child=end_plank,
                    parent_xform=_xf(ramp_position + np.array([-x_sign * 0.8, y, 0.1])),
                    child_xform=_xf((x_sign * 0.5 * plank_length, y, 0.0)),
                )

        for side_index, y_sign in enumerate((-1.0, 1.0)):
            for plank_index in range(plank_count):
                plank_x = -0.5 * span + (plank_index + 0.5) * plank_length
                cable_index = int(np.argmin(np.abs(cable_x_positions[side_index] - plank_x)))
                cable_midpoint = cable_midpoints[side_index][cable_index]
                bottom = np.array([plank_x, y_sign * 0.46 * plank_width, self.deck_z + 0.5 * plank_thickness])
                height = cable_midpoint[2] - bottom[2]
                if height < 0.3:
                    continue
                hanger_count = max(1, min(31, int(round(height / 0.55))))
                top = np.array([cable_midpoint[0], y_sign * cable_y, cable_midpoint[2]])
                hanger_points = [top + (bottom - top) * (index / hanger_count) for index in range(hanger_count + 1)]
                self._add_rope(
                    builder,
                    hanger_points,
                    0.045,
                    2.0,
                    (cable_links[side_index][cable_index], np.zeros(3)),
                    (
                        self.planks[plank_index],
                        np.array([0.0, y_sign * 0.46 * plank_width, 0.5 * plank_thickness]),
                    ),
                )

        self.cargo: list[int] = []
        for index in range(cargo_count):
            x = (index - 1.5) * span * 0.12
            y = ((index & 1) - 0.5) * plank_width * 0.4
            cargo = self._add_box(
                builder,
                None,
                (0.8, 0.8, 0.8),
                (x, y, self.deck_z + 3.0 + index * 1.5),
                density=1.5,
                color=CARGO_COLOR,
            )
            self.cargo.append(cargo)

        return self._extents

    def _shape_cfg(
        self,
        density: float,
        collision_group: int = 1,
        collide: bool = True,
    ) -> newton.ModelBuilder.ShapeConfig:
        return newton.ModelBuilder.ShapeConfig(
            density=density,
            collision_group=collision_group,
            has_shape_collision=collide,
            mu=self.default_friction,
        )

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
        collision_group: int = 1,
        color=DECK_COLOR,
    ) -> int:
        half_extents = tuple(0.5 * float(value) for value in full_extents)
        cfg = self._shape_cfg(density, collision_group)
        if body is None:
            body = builder.add_link(xform=_xf(position))
            xform = None
        else:
            xform = _xf(position) if body == -1 else None
        builder.add_shape_box(
            body,
            xform=xform,
            hx=half_extents[0],
            hy=half_extents[1],
            hz=half_extents[2],
            cfg=cfg,
            color=color,
        )
        if body >= 0:
            self._record_extent(body, default_box_half_extents(*half_extents))
        return body

    def _add_rope(self, builder: newton.ModelBuilder, points, radius, density, pin_start, pin_end):
        cfg = self._shape_cfg(density, collision_group=0, collide=False)
        links: list[int] = []
        midpoints: list[np.ndarray] = []
        previous_body = None
        previous_far_anchor = None
        for point_a, point_b in pairwise(points):
            direction = point_b - point_a
            length = float(np.linalg.norm(direction))
            if length < 1.0e-6:
                continue
            midpoint = 0.5 * (point_a + point_b)
            body = builder.add_link(
                xform=wp.transform(p=wp.vec3(*midpoint), q=_quat_z_to_dir(direction)),
            )
            half_height = 0.4 * length
            builder.add_shape_capsule(
                body,
                radius=radius,
                half_height=half_height,
                cfg=cfg,
                color=CABLE_COLOR,
            )
            self._record_extent(body, default_capsule_half_extents(radius, half_height))

            near_anchor = _xf((0.0, 0.0, -0.5 * length))
            if previous_body is None:
                pin_body, pin_anchor = pin_start
                parent_xform = _xf(points[0]) if pin_body == -1 else _xf(pin_anchor)
                builder.add_joint_ball(
                    parent=pin_body,
                    child=body,
                    parent_xform=parent_xform,
                    child_xform=near_anchor,
                )
            else:
                builder.add_joint_ball(
                    parent=previous_body,
                    child=body,
                    parent_xform=_xf(previous_far_anchor),
                    child_xform=near_anchor,
                )

            links.append(body)
            midpoints.append(midpoint)
            previous_body = body
            previous_far_anchor = (0.0, 0.0, 0.5 * length)

        if previous_body is not None:
            pin_body, pin_anchor = pin_end
            parent_xform = _xf(points[-1]) if pin_body == -1 else _xf(pin_anchor)
            builder.add_joint_ball(
                parent=pin_body,
                child=previous_body,
                parent_xform=parent_xform,
                child_xform=_xf(previous_far_anchor),
            )
        return links, midpoints

    def configure_camera(self, viewer) -> None:
        viewer.set_camera(pos=wp.vec3(25.0, -25.0, 11.0), pitch=-14.0, yaw=135.0)

    def test_final(self) -> None:
        super().test_final()
        positions = self.bodies.position.numpy()
        plank_z = positions[np.asarray(self.planks) + 1, 2]
        cargo_z = positions[np.asarray(self.cargo) + 1, 2]
        if not ((plank_z > 1.0).all() and (plank_z < 4.5).all()):
            raise AssertionError(f"bridge deck left its hanging envelope: z={plank_z}")
        if (cargo_z < 0.25).any():
            raise AssertionError(f"bridge cargo escaped below the scene: z={cargo_z}")


if __name__ == "__main__":
    run_ported_example(Example)
