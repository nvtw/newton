# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX slinky stairs
#
# A capsule-segment helical spring walks down a staircase. Cable joints
# preserve the helix's elastic rest curvature while allowing it to stretch,
# bend, twist, and collide with non-neighboring coils. The upper half starts
# with a graded forward/downward velocity and pitch rate, initiating the
# familiar end-over-end slinky motion without a scripted actuator.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_slinky_stairs
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_capsule_half_extents,
    run_ported_example,
)

NUM_STEPS = 8
STEP_TREAD = 0.42
STEP_RISE = 0.20
STAIR_WIDTH = 1.60
TOP_LANDING_DEPTH = 1.20

SLINKY_RADIUS = 0.23
WIRE_RADIUS = 0.018
NUM_TURNS = 12
SEGMENTS_PER_TURN = 8
TURN_PITCH = 0.043
SLINKY_DENSITY = 180.0

STRETCH_STIFFNESS = 2.0e6
STRETCH_DAMPING = 30.0
SHEAR_STIFFNESS = 2.0e6
SHEAR_DAMPING = 30.0
BEND_STIFFNESS = 0.25
BEND_DAMPING = 0.025
TWIST_STIFFNESS = 0.10
TWIST_DAMPING = 0.02

KICK_FORWARD_SPEED = 2.00
KICK_DOWN_SPEED = 0.50
KICK_PITCH_RATE = 6.0

SLINKY_COLOR = (0.92, 0.50, 0.04)
STAIR_COLOR = (0.42, 0.45, 0.50)
GROUND_COLOR = (0.30, 0.32, 0.35)


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


class Example(PortedExample):
    """Simulate an elastic slinky walking down rigid stairs."""

    fps = 60
    sim_substeps = 10
    solver_iterations = 6
    velocity_iterations = 1
    default_friction = 0.65
    broad_phase = "sap"
    step_layout = "single_world"
    shape_pairs_max = 32768
    show_contacts = False
    evaluate_fk = False
    step_report_label = "SlinkyStairs"

    def build_scene(self, builder: newton.ModelBuilder):
        """Build the staircase and initialize the helical spring."""
        builder.default_shape_cfg.gap = 0.004
        builder.add_ground_plane(height=0.0, color=GROUND_COLOR)
        self._add_stairs(builder)

        points = self._slinky_points()
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=SLINKY_DENSITY,
            mu=self.default_friction,
            restitution=0.0,
            gap=0.003,
        )
        bodies, _ = builder.add_rod(
            positions=[wp.vec3(*point) for point in points],
            quaternions=None,
            radius=WIRE_RADIUS,
            cfg=shape_cfg,
            stretch_stiffness=STRETCH_STIFFNESS,
            stretch_damping=STRETCH_DAMPING,
            shear_stiffness=SHEAR_STIFFNESS,
            shear_damping=SHEAR_DAMPING,
            bend_stiffness=BEND_STIFFNESS,
            bend_damping=BEND_DAMPING,
            twist_stiffness=TWIST_STIFFNESS,
            twist_damping=TWIST_DAMPING,
            label="slinky",
            color=SLINKY_COLOR,
            body_frame_origin="com",
        )
        self.slinky_bodies = [int(body) for body in bodies]
        self.initial_centroid_x = float(np.mean(points[:, 0]))
        self.initial_min_z = float(np.min(points[:, 2]) - WIRE_RADIUS)

        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        extents = [default_capsule_half_extents(WIRE_RADIUS, 0.5 * float(length)) for length in segment_lengths]

        # A graded kick folds the upper coils over the stair edge. Lower coils
        # remain nearly stationary, so cable tension drives the first transfer.
        count = len(self.slinky_bodies)
        for index, body in enumerate(self.slinky_bodies):
            height_fraction = (index + 0.5) / count
            kick = _smoothstep((height_fraction - 0.38) / 0.62)
            builder.body_qd[body] = wp.spatial_vector(
                KICK_FORWARD_SPEED * kick,
                0.0,
                -KICK_DOWN_SPEED * kick,
                0.0,
                KICK_PITCH_RATE * kick,
                0.0,
            )

        return extents

    def _add_stairs(self, builder: newton.ModelBuilder) -> None:
        """Add a top landing followed by descending solid steps."""
        top_height = NUM_STEPS * STEP_RISE
        static_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=self.default_friction,
            restitution=0.0,
            gap=builder.default_shape_cfg.gap,
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(
                wp.vec3(-0.5 * TOP_LANDING_DEPTH, 0.0, 0.5 * top_height),
                wp.quat_identity(),
            ),
            hx=0.5 * TOP_LANDING_DEPTH,
            hy=0.5 * STAIR_WIDTH,
            hz=0.5 * top_height,
            cfg=static_cfg,
            color=STAIR_COLOR,
            label="top_landing",
        )
        for index in range(1, NUM_STEPS):
            surface_height = (NUM_STEPS - index) * STEP_RISE
            builder.add_shape_box(
                -1,
                xform=wp.transform(
                    wp.vec3((index - 0.5) * STEP_TREAD, 0.0, 0.5 * surface_height),
                    wp.quat_identity(),
                ),
                hx=0.5 * STEP_TREAD,
                hy=0.5 * STAIR_WIDTH,
                hz=0.5 * surface_height,
                cfg=static_cfg,
                color=STAIR_COLOR,
                label=f"step_{index}",
            )

    def _slinky_points(self) -> np.ndarray:
        """Return centerline nodes for a compressed upright helix."""
        segment_count = NUM_TURNS * SEGMENTS_PER_TURN
        top_height = NUM_STEPS * STEP_RISE
        base_z = top_height + WIRE_RADIUS + 0.003
        center_x = -SLINKY_RADIUS - 0.015
        points = np.empty((segment_count + 1, 3), dtype=np.float32)
        for index in range(segment_count + 1):
            turns = index / SEGMENTS_PER_TURN
            angle = 2.0 * math.pi * turns
            points[index] = (
                center_x + SLINKY_RADIUS * math.cos(angle),
                SLINKY_RADIUS * math.sin(angle),
                base_z + TURN_PITCH * turns,
            )
        return points

    def configure_camera(self, viewer) -> None:
        """Frame the complete stair flight and initial slinky pose."""
        viewer.set_camera(
            pos=wp.vec3(2.8, -5.2, 2.7),
            pitch=-12.0,
            yaw=120.0,
        )

    def test_final(self) -> None:
        """Verify the slinky remains finite and starts descending."""
        super().test_final()
        body_q = self.state.body_q.numpy()[self.slinky_bodies]
        centroid_x = float(np.mean(body_q[:, 0]))
        min_z = float(np.min(body_q[:, 2]) - WIRE_RADIUS)
        if centroid_x <= self.initial_centroid_x + 0.10 or min_z >= self.initial_min_z - 0.5 * STEP_RISE:
            raise AssertionError(
                "slinky did not begin descending "
                f"(centroid_dx={centroid_x - self.initial_centroid_x:.3f} m, "
                f"min_dz={min_z - self.initial_min_z:.3f} m)"
            )


def _configure_parser(parser) -> None:
    parser.set_defaults(viewer="optix", num_frames=360)


if __name__ == "__main__":
    run_ported_example(Example, _configure_parser)
