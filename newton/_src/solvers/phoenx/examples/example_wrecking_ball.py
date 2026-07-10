# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX wrecking ball.
#
# Port of the PEEL "WreckingBall" solver stress test. An open dynamic
# building frame with furnished floors is smashed by a giant ball on a
# ball-jointed capsule chain hung from a static crane.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_wrecking_ball
#      python -m newton._src.solvers.phoenx.examples.example_wrecking_ball --fix-ball-attachment
###########################################################################

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    default_capsule_half_extents,
    default_sphere_half_extents,
)

BUILDING_COLOR = (0.68, 0.70, 0.72)
FURNITURE_COLOR = (0.58, 0.40, 0.26)
CRANE_COLOR = (0.44, 0.46, 0.49)
CHAIN_COLOR = (0.16, 0.18, 0.20)
BALL_COLOR = (0.12, 0.13, 0.15)


def _p(*v) -> np.ndarray:
    return np.array(v, dtype=float)


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


def _xf(position, q: wp.quat | None = None) -> wp.transform:
    return wp.transform(
        p=wp.vec3(float(position[0]), float(position[1]), float(position[2])),
        q=q or wp.quat_identity(),
    )


def _yaw(angle: float) -> wp.quat:
    if angle == 0.0:
        return wp.quat_identity()
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)


class Example(PortedExample):
    # The original PEEL scene deliberately spawns the ball with a large
    # chain-end violation. Keep that available for solver A/B testing,
    # but the CLI can opt into a spawn-consistent attachment.
    FIX_BALL_ATTACHMENT = False

    sim_substeps = 20
    solver_iterations = 12
    velocity_iterations = 1
    step_layout = "single_world"
    broad_phase = "sap"
    shape_pairs_max = 200000
    default_friction = 0.7
    show_contacts = False
    finalize_kwargs: ClassVar[dict[str, bool]] = {"skip_validation_joints": True}
    evaluate_fk = False

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane(color=(0.70, 0.72, 0.74))
        self._extents: list[tuple[float, float, float] | None] = []
        self.structure: list[int] = []

        self.n_floors = 8
        width, depth, floor_height, slab_thickness, column_size = 14.0, 10.0, 2.7, 0.3, 0.55
        column_height = floor_height - slab_thickness

        for floor in range(self.n_floors):
            floor_z = floor * floor_height
            for grid_x in range(3):
                for grid_y in range(2):
                    self.structure.append(
                        self._box(
                            builder,
                            None,
                            (column_size, column_size, column_height),
                            (-0.5 * width + grid_x * 0.5 * width, -0.5 * depth + grid_y * depth, floor_z + 0.5 * column_height),
                            1.6,
                        )
                    )
            for x in (-0.25 * width, 0.25 * width):
                for y in (-0.5 * depth, 0.5 * depth):
                    self.structure.append(
                        self._box(builder, None, (column_size, column_size, column_height), (x, y, floor_z + 0.5 * column_height), 1.6)
                    )
            self.structure.append(
                self._box(
                    builder,
                    None,
                    (width + 0.8, depth + 0.8, slab_thickness),
                    (0.0, 0.0, floor_z + floor_height - 0.5 * slab_thickness),
                    0.55,
                )
            )

        for floor in range(self.n_floors):
            self._furnish_floor(builder, floor, floor * floor_height)

        jib_tip_x = 8.5
        jib_z = self.n_floors * floor_height + 4.6
        ball_radius = 1.9
        chain_length = jib_z - self.n_floors * floor_height * 0.5 - ball_radius
        theta = 1.1
        mast_x = jib_tip_x + math.sin(theta) * (chain_length + ball_radius) + ball_radius + 1.2
        self._box(builder, -1, (5.5, 5.5, 0.8), (mast_x + 1.0, 0.0, 0.4), color=CRANE_COLOR)
        self._box(builder, -1, (1.1, 1.1, jib_z), (mast_x, 0.0, 0.5 * jib_z), color=CRANE_COLOR)
        self._box(builder, -1, (mast_x - jib_tip_x + 2.4, 0.9, 0.9), (0.5 * (mast_x + jib_tip_x), 0.0, jib_z + 0.45), color=CRANE_COLOR)
        self._box(builder, -1, (2.0, 2.2, 1.6), (mast_x + 2.6, 0.0, jib_z - 0.5), color=CRANE_COLOR)

        tip = _p(jib_tip_x, 0.0, jib_z)
        hang = chain_length + ball_radius
        ball_center = tip + _p(math.sin(theta) * hang, 0.0, -math.cos(theta) * hang)
        self.ball = builder.add_link(xform=_xf(ball_center))
        builder.add_shape_sphere(self.ball, radius=ball_radius, cfg=self._cfg(9.0), color=BALL_COLOR)
        self._record_extent(self.ball, default_sphere_half_extents(ball_radius))
        self.ball_start = ball_center.copy()

        direction = ball_center - tip
        direction_unit = direction / np.linalg.norm(direction)
        chain_count = max(3, min(60, int(chain_length / 0.55 + 0.5)))
        chain_points = [tip + direction_unit * (chain_length * index / chain_count) for index in range(chain_count + 1)]
        fix_arg = getattr(self.args, "fix_ball_attachment", None)
        use_fix = self.FIX_BALL_ATTACHMENT if fix_arg is None else bool(fix_arg)
        ball_anchor = -direction_unit * ball_radius if use_fix else _p(0.0, 0.0, ball_radius)
        self._rope(builder, chain_points, 0.07, 6.0, (-1, None), (self.ball, ball_anchor))

        return self._extents

    def post_build(self) -> None:
        self.struct_pos0 = self.model.body_q.numpy()[self.structure, :3].copy()

    def _cfg(
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
        if body < 0:
            return
        while len(self._extents) <= body:
            self._extents.append(None)
        current = self._extents[body]
        if current is None:
            self._extents[body] = extent
        else:
            self._extents[body] = tuple(max(current[i], extent[i]) for i in range(3))

    def _box(
        self,
        builder: newton.ModelBuilder,
        body: int | None,
        full_extents,
        position,
        density: float = 0.0,
        collision_group: int = 1,
        yaw: float = 0.0,
        color=BUILDING_COLOR,
    ) -> int:
        half_extents = tuple(0.5 * float(value) for value in full_extents)
        cfg = self._cfg(density, collision_group)
        if body is None:
            body = builder.add_link(xform=_xf(position, _yaw(yaw)))
            xform = None
        else:
            xform = _xf(position, _yaw(yaw)) if body == -1 else _xf(position)
        builder.add_shape_box(
            body,
            xform=xform,
            hx=half_extents[0],
            hy=half_extents[1],
            hz=half_extents[2],
            cfg=cfg,
            color=color,
        )
        self._record_extent(body, default_box_half_extents(*half_extents))
        return body

    def _rope(self, builder: newton.ModelBuilder, points, radius, density, pin_start, pin_end) -> None:
        cfg = self._cfg(density, collision_group=-1)
        previous_body = None
        previous_far_anchor = None
        for point_a, point_b in zip(points[:-1], points[1:]):
            segment = point_b - point_a
            length = float(np.linalg.norm(segment))
            if length < 1.0e-6:
                continue
            midpoint = 0.5 * (point_a + point_b)
            body = builder.add_link(
                xform=wp.transform(p=wp.vec3(*midpoint), q=_quat_z_to_dir(segment)),
            )
            half_height = 0.4 * length
            builder.add_shape_capsule(body, radius=radius, half_height=half_height, cfg=cfg, color=CHAIN_COLOR)
            self._record_extent(body, default_capsule_half_extents(radius, half_height))

            near_anchor = _xf((0.0, 0.0, -0.5 * length))
            if previous_body is None:
                pin_body, pin_anchor = pin_start
                parent_xform = _xf(points[0]) if pin_body == -1 else _xf(pin_anchor)
                builder.add_joint_ball(parent=pin_body, child=body, parent_xform=parent_xform, child_xform=near_anchor)
            else:
                builder.add_joint_ball(
                    parent=previous_body,
                    child=body,
                    parent_xform=_xf(previous_far_anchor),
                    child_xform=near_anchor,
                )
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

    def _furniture(self, builder: newton.ModelBuilder, parts, position, yaw: float = 0.0) -> int:
        body = builder.add_link(xform=_xf(position, _yaw(yaw)))
        max_half = np.zeros(3)
        for size, offset, density in parts:
            half = 0.5 * np.asarray(size, dtype=float)
            offset = np.asarray(offset, dtype=float)
            builder.add_shape_box(
                body,
                xform=_xf(offset),
                hx=float(half[0]),
                hy=float(half[1]),
                hz=float(half[2]),
                cfg=self._cfg(density),
                color=FURNITURE_COLOR,
            )
            max_half = np.maximum(max_half, np.abs(offset) + half)
        self._record_extent(body, tuple(float(value) for value in max_half))
        return body

    def _sofa(self, builder: newton.ModelBuilder, position, yaw: float) -> None:
        self._furniture(
            builder,
            [
                (_p(2.0, 0.85, 0.4), _p(0.0, 0.0, 0.2), 0.45),
                (_p(2.0, 0.25, 0.55), _p(0.0, -0.32, 0.65), 0.45),
                (_p(0.25, 0.85, 0.3), _p(-0.93, 0.0, 0.55), 0.45),
                (_p(0.25, 0.85, 0.3), _p(0.93, 0.0, 0.55), 0.45),
            ],
            position,
            yaw,
        )

    def _table(self, builder: newton.ModelBuilder, position, top, height: float) -> None:
        parts = [(_p(*top), _p(0.0, 0.0, height - 0.5 * top[2]), 0.6)]
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                parts.append(
                    (
                        _p(0.09, 0.09, height - top[2]),
                        _p(x_sign * (0.5 * top[0] - 0.1), y_sign * (0.5 * top[1] - 0.1), 0.5 * (height - top[2])),
                        0.6,
                    )
                )
        self._furniture(builder, parts, position)

    def _chair(self, builder: newton.ModelBuilder, position, yaw: float) -> None:
        self._furniture(
            builder,
            [
                (_p(0.45, 0.45, 0.08), _p(0.0, 0.0, 0.45), 0.45),
                (_p(0.38, 0.38, 0.41), _p(0.0, 0.0, 0.205), 0.25),
                (_p(0.45, 0.07, 0.5), _p(0.0, -0.19, 0.74), 0.45),
            ],
            position,
            yaw,
        )

    def _bed(self, builder: newton.ModelBuilder, position, yaw: float) -> None:
        self._furniture(
            builder,
            [
                (_p(2.0, 1.5, 0.35), _p(0.0, 0.0, 0.175), 0.5),
                (_p(1.9, 1.4, 0.2), _p(0.0, 0.0, 0.45), 0.2),
                (_p(0.12, 1.5, 0.75), _p(-1.0, 0.0, 0.375), 0.5),
            ],
            position,
            yaw,
        )

    def _toilet(self, builder: newton.ModelBuilder, position, yaw: float) -> None:
        self._furniture(
            builder,
            [
                (_p(0.45, 0.55, 0.42), _p(0.0, 0.0, 0.21), 0.8),
                (_p(0.45, 0.18, 0.5), _p(0.0, -0.32, 0.62), 0.8),
            ],
            position,
            yaw,
        )

    def _tv(self, builder: newton.ModelBuilder, position, yaw: float) -> None:
        self._furniture(
            builder,
            [
                (_p(1.5, 0.45, 0.5), _p(0.0, 0.0, 0.25), 0.6),
                (_p(1.3, 0.08, 0.75), _p(0.0, 0.0, 0.9), 0.3),
            ],
            position,
            yaw,
        )

    def _fridge(self, builder: newton.ModelBuilder, position) -> None:
        self._box(builder, None, (0.75, 0.75, 1.85), position + _p(0.0, 0.0, 0.925), 0.8, color=FURNITURE_COLOR)

    def _counter(self, builder: newton.ModelBuilder, position, length: float, yaw: float) -> None:
        self._box(builder, None, (length, 0.65, 0.95), position + _p(0.0, 0.0, 0.475), 0.7, yaw=yaw, color=FURNITURE_COLOR)

    def _shelf(self, builder: newton.ModelBuilder, position, yaw: float) -> None:
        self._box(builder, None, (1.2, 0.35, 1.7), position + _p(0.0, 0.0, 0.85), 0.5, yaw=yaw, color=FURNITURE_COLOR)

    def _dining_set(self, builder: newton.ModelBuilder, position) -> None:
        self._table(builder, position, (1.7, 1.0, 0.08), 0.74)
        self._chair(builder, position + _p(0.0, 0.85, 0.0), math.pi)
        self._chair(builder, position + _p(0.0, -0.85, 0.0), 0.0)
        self._chair(builder, position + _p(1.15, 0.0, 0.0), 0.5 * math.pi)
        self._chair(builder, position + _p(-1.15, 0.0, 0.0), -0.5 * math.pi)

    def _furnish_floor(self, builder: newton.ModelBuilder, floor: int, floor_z: float) -> None:
        if floor % 3 == 0:
            self._counter(builder, _p(-5.8, -3.5, floor_z), 3.4, 0.0)
            self._fridge(builder, _p(-3.6, -4.2, floor_z))
            self._counter(builder, _p(-6.3, -1.2, floor_z), 2.0, 0.5 * math.pi)
            self._dining_set(builder, _p(-1.5, -2.2, floor_z))
            self._sofa(builder, _p(3.5, 3.2, floor_z), math.pi)
            self._tv(builder, _p(3.5, -0.5, floor_z), 0.0)
            self._shelf(builder, _p(6.2, 2.0, floor_z), 0.5 * math.pi)
        elif floor % 3 == 1:
            self._bed(builder, _p(-4.5, 2.5, floor_z), 0.0)
            self._shelf(builder, _p(-6.4, -1.0, floor_z), 0.5 * math.pi)
            self._sofa(builder, _p(0.5, -3.3, floor_z), 0.0)
            self._table(builder, _p(0.5, -1.2, floor_z), (0.9, 0.9, 0.07), 0.45)
            self._toilet(builder, _p(5.8, -3.8, floor_z), 0.5 * math.pi)
            self._counter(builder, _p(5.8, -1.8, floor_z), 1.4, 0.5 * math.pi)
            self._bed(builder, _p(4.5, 2.8, floor_z), 0.5 * math.pi)
        else:
            self._table(builder, _p(-4.5, -3.0, floor_z), (1.6, 0.8, 0.07), 0.74)
            self._chair(builder, _p(-4.5, -1.9, floor_z), math.pi)
            self._shelf(builder, _p(-6.4, 0.5, floor_z), 0.5 * math.pi)
            self._shelf(builder, _p(-6.4, 2.5, floor_z), 0.5 * math.pi)
            self._sofa(builder, _p(1.0, 2.8, floor_z), math.pi)
            self._sofa(builder, _p(-1.5, 0.0, floor_z), -0.5 * math.pi)
            self._tv(builder, _p(3.0, -3.5, floor_z), 0.0)
            self._dining_set(builder, _p(4.5, 1.5, floor_z))

    def configure_camera(self, viewer) -> None:
        viewer.set_camera(pos=wp.vec3(42.0, -36.0, 22.0), pitch=-18.0, yaw=132.0)

    def test_post_step(self) -> None:
        speeds = np.linalg.norm(self.state.body_qd.numpy()[:, 3:6], axis=1)
        self.peak_speed = max(getattr(self, "peak_speed", 0.0), float(speeds.max()))

    def test_final(self) -> None:
        super().test_final()
        body_q = self.state.body_q.numpy()
        body_qd = self.state.body_qd.numpy()
        ball_position = body_q[self.ball, :3]
        swung = self.ball_start[0] - ball_position[0]
        drift = np.linalg.norm(body_q[self.structure, :3] - self.struct_pos0, axis=1)
        final_max_speed = float(np.linalg.norm(body_qd[:, 3:6], axis=1).max())
        print(f"[PhoenX wrecking] ball pos = {np.round(ball_position, 2)} (swung {swung:.2f} m toward building)")
        print(
            f"[PhoenX wrecking] peak speed = {getattr(self, 'peak_speed', float('nan')):.1f} m/s "
            f"final max = {final_max_speed:.2f} m/s"
        )
        print(f"[PhoenX wrecking] structure displaced >0.5 m: {int((drift > 0.5).sum())}/{len(self.structure)} bodies")


def run_example() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        choices=("classic", "jacobi"),
        default="classic",
        help="Select graph-colored PGS or the uncolored scalar-row Jacobi solver.",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=10,
        help="Estimated classic color count used by Jacobi.",
    )
    parser.add_argument(
        "--fix-ball-attachment",
        action="store_true",
        help="Anchor the chain at the spawn-consistent point, removing the deliberate initial constraint violation.",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    run_example()
