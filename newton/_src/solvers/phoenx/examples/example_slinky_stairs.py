# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX slinky stairs
#
# A capsule-segment helical spring walks down a staircase. Rod joints
# preserve the helix's elastic rest curvature while allowing it to stretch,
# bend, twist, and collide with non-neighboring coils. The undeformed spring
# starts with a coherent tipping velocity about the first tread edge.
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

NUM_STEPS = 5
STEP_TREAD = 0.90
STEP_RISE = 0.65
STAIR_WIDTH = 2.00
TOP_LANDING_DEPTH = 1.80

SLINKY_RADIUS = 0.23
WIRE_RADIUS = 0.018
NUM_TURNS = 25
SEGMENTS_PER_TURN = 20
TURN_PITCH = 0.046
SLINKY_DENSITY = 1200.0
SLINKY_CONTACT_GAP = 0.005

REFERENCE_YOUNGS_MODULUS = 3.6e9
REFERENCE_SHEAR_MODULUS = 2.4e9
MATERIAL_STIFFNESS_SCALE = 0.60
STRETCH_DAMPING = 30.0
SHEAR_DAMPING = 30.0
BEND_DAMPING = 0.25
TWIST_DAMPING = 0.25

INITIAL_CENTER_X = -0.45
INITIAL_HEIGHT_OFFSET = 0.08
INITIAL_FORWARD_SPEED = 1.0

SLINKY_COLOR = (0.92, 0.50, 0.04)
STAIR_COLOR = (0.42, 0.45, 0.50)
GROUND_COLOR = (0.30, 0.32, 0.35)


class Example(PortedExample):
    """Simulate an elastic slinky walking down rigid stairs."""

    fps = 60
    sim_substeps = 10
    solver_iterations = 6
    velocity_iterations = 1
    default_friction = 0.40
    broad_phase = "sap"
    step_layout = "single_world"
    # Direct-contact runs otherwise apply one articulation-wide response per
    # contact. Grouped mass splitting processes independent contact groups in
    # parallel and restores the exact joint manifold between grouped sweeps.
    mass_splitting = True
    mass_splitting_color_group_size = 3
    max_colored_partitions = 8
    shape_pairs_max = 32768
    speculative_contact_gap_max = 0.05
    show_contacts = True
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
            gap=SLINKY_CONTACT_GAP,
        )
        rod = newton.Rod(
            points,
            radius=WIRE_RADIUS,
            youngs_modulus=MATERIAL_STIFFNESS_SCALE * REFERENCE_YOUNGS_MODULUS,
            shear_modulus=MATERIAL_STIFFNESS_SCALE * REFERENCE_SHEAR_MODULUS,
        )
        bodies, _ = builder.add_rod(
            rod=rod,
            cfg=shape_cfg,
            stretch_damping=STRETCH_DAMPING,
            shear_damping=SHEAR_DAMPING,
            bend_damping=BEND_DAMPING,
            twist_damping=TWIST_DAMPING,
            label="slinky",
            color=SLINKY_COLOR,
            body_frame_origin="com",
        )
        self.slinky_bodies = [int(body) for body in bodies]
        centers = 0.5 * (points[:-1] + points[1:])
        linear_velocity = np.asarray((INITIAL_FORWARD_SPEED, 0.0, 0.0), dtype=np.float32)
        angular_velocity = np.zeros(3, dtype=np.float32)
        for body in self.slinky_bodies:
            builder.body_qd[body] = wp.spatial_vector(*linear_velocity, *angular_velocity)

        self.initial_centroid_x = float(np.mean(centers[:, 0]))
        self.initial_min_x = float(np.min(centers[:, 0]) - WIRE_RADIUS)
        self.initial_min_z = float(np.min(centers[:, 2]) - WIRE_RADIUS)

        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        extents = [default_capsule_half_extents(WIRE_RADIUS, 0.5 * float(length)) for length in segment_lengths]

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
        """Return a compressed helix whose axis points down the stair flight."""
        segment_count = NUM_TURNS * SEGMENTS_PER_TURN
        top_height = NUM_STEPS * STEP_RISE
        center_z = top_height + SLINKY_RADIUS + WIRE_RADIUS + 0.003 + INITIAL_HEIGHT_OFFSET
        start_x = INITIAL_CENTER_X - 0.5 * NUM_TURNS * TURN_PITCH
        points = np.empty((segment_count + 1, 3), dtype=np.float32)
        for index in range(segment_count + 1):
            turns = index / SEGMENTS_PER_TURN
            angle = 2.0 * math.pi * turns
            points[index] = (
                start_x + TURN_PITCH * turns,
                SLINKY_RADIUS * math.cos(angle),
                center_z + SLINKY_RADIUS * math.sin(angle),
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
        min_x = float(np.min(body_q[:, 0]) - WIRE_RADIUS)
        min_z = float(np.min(body_q[:, 2]) - WIRE_RADIUS)
        if (
            centroid_x <= self.initial_centroid_x + 4.0 * STEP_TREAD
            or min_x <= self.initial_min_x + 3.0 * STEP_TREAD
            or min_z >= self.initial_min_z - (NUM_STEPS - 1) * STEP_RISE
        ):
            raise AssertionError(
                "slinky did not carry its complete span down the stairs "
                f"(centroid_dx={centroid_x - self.initial_centroid_x:.3f} m, "
                f"trailing_dx={min_x - self.initial_min_x:.3f} m, "
                f"min_dz={min_z - self.initial_min_z:.3f} m)"
            )


def _configure_parser(parser) -> None:
    parser.set_defaults(viewer="optix", num_frames=600)


if __name__ == "__main__":
    run_ported_example(Example, _configure_parser)
