# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX twisted rod
#
# A stress-free elastic rod hangs between two rotating support cubes. The
# left cube rotates around the rod tangent, building torsion until the rod
# buckles into loops. Collision-free material stripes expose local frame
# rotation while the capsule centerline supplies self-collision.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_twisted_rod
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

ROD_RADIUS = 0.018
ROD_DENSITY = 1100.0
ROD_YOUNGS_MODULUS = 0.2e9
ROD_SHEAR_MODULUS = 0.1e9
ROD_BEND_RIGIDITY_SCALE = 0.50

NUM_SEGMENTS = 400
ROD_SPAN = 6.0
ROD_SAG = 1.10
SUPPORT_HALF_EXTENT = 0.14
DRIVE_DURATION = 20.0
SUPPORT_TWIST_RATE = 20.0 * 2.0 * math.pi / DRIVE_DURATION
SUPPORT_APPROACH_DISTANCE = 4.2
SUPPORT_APPROACH_RATE = SUPPORT_APPROACH_DISTANCE / DRIVE_DURATION
STRIPE_HALF_WIDTH = 0.004
STRIPE_HALF_THICKNESS = 0.0025

ROD_COLOR = (0.90, 0.82, 0.25)
STRIPE_COLOR = (0.30, 0.58, 0.63)
SUPPORT_COLORS = ((0.76, 0.68, 0.58), (0.58, 0.68, 0.76))


@wp.kernel(enable_backward=False)
def _rotate_supports_kernel(
    body_indices: wp.array[wp.int32],
    twist_rates: wp.array[wp.float32],
    approach_rates: wp.array[wp.float32],
    target_x: wp.array[wp.float32],
    dt: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    support = wp.tid()
    body = body_indices[support]
    pose = body_q[body]
    position = wp.transform_get_translation(pose)
    approach_speed = approach_rates[support]
    translation_dt = wp.float32(0.0)
    if approach_speed > 0.0:
        translation_dt = wp.clamp((target_x[support] - position[0]) / approach_speed, 0.0, dt)
    linear_velocity = wp.vec3(approach_speed, 0.0, 0.0)
    position += linear_velocity * translation_dt
    rotation = wp.transform_get_rotation(pose)
    axis = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
    angular_speed = twist_rates[support]
    delta = wp.quat_from_axis_angle(axis, angular_speed * dt)
    body_q[body] = wp.transform(position, wp.normalize(wp.mul(delta, rotation)))
    if translation_dt == 0.0:
        linear_velocity = wp.vec3()
    body_qd[body] = wp.spatial_vector(linear_velocity, axis * angular_speed)


def _hanging_centerline() -> np.ndarray:
    """Return a smooth, untwisted centerline hanging between supports."""
    parameter = np.linspace(-1.0, 1.0, NUM_SEGMENTS + 1, dtype=np.float64)
    points = np.empty((NUM_SEGMENTS + 1, 3), dtype=np.float32)
    points[:, 0] = 0.5 * ROD_SPAN * parameter
    # A small out-of-plane imperfection selects a repeatable torsional
    # buckling direction without adding initial material twist.
    points[:, 1] = 0.008 * np.sin(math.pi * (parameter + 1.0))
    points[:, 2] = 1.45 - ROD_SAG * (1.0 - parameter * parameter)
    return points


class Example(PortedExample):
    """Twist a freely hanging rod by rotating its left support cube."""

    fps = 60
    collision_updates_per_frame = 2
    sim_substeps = 15
    solver_iterations = 8
    velocity_iterations = 1
    default_friction = 0.5
    broad_phase = "sap"
    step_layout = "single_world"
    mass_splitting = True
    mass_splitting_color_group_size = 3
    max_colored_partitions = 8
    shape_pairs_max = 32768
    speculative_contact_gap_max = 0.05
    show_contacts = True
    evaluate_fk = False
    step_report_label = "TwistedRod"

    def build_scene(self, builder: newton.ModelBuilder):
        """Build an initially stress-free hanging rod and driven supports."""
        points = _hanging_centerline()
        section_area = math.pi * ROD_RADIUS**2
        area_moment = 0.25 * math.pi * ROD_RADIUS**4
        polar_moment = 2.0 * area_moment
        rod = newton.Rod(
            points,
            radius=ROD_RADIUS,
            stretch_rigidity=ROD_YOUNGS_MODULUS * section_area,
            shear_rigidity=0.9 * ROD_SHEAR_MODULUS * section_area,
            bend_rigidity=ROD_BEND_RIGIDITY_SCALE * ROD_YOUNGS_MODULUS * area_moment,
            twist_rigidity=ROD_SHEAR_MODULUS * polar_moment,
        )
        rod_cfg = newton.ModelBuilder.ShapeConfig(
            density=ROD_DENSITY,
            mu=self.default_friction,
            restitution=0.0,
            gap=0.002,
        )
        bodies, _ = builder.add_rod(
            rod=rod,
            cfg=rod_cfg,
            stretch_damping=2.0,
            shear_damping=2.0,
            bend_damping=0.25,
            twist_damping=0.10,
            label="twisted_rod",
            wrap_in_articulation=True,
            body_frame_origin="com",
            color=ROD_COLOR,
        )
        self.rod_bodies = [int(body) for body in bodies]
        self.anchor_bodies = (self.rod_bodies[0], self.rod_bodies[-1])

        for body in self.anchor_bodies:
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        visual_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
        for index, (body, segment_length) in enumerate(zip(self.rod_bodies, segment_lengths, strict=True)):
            builder.add_shape_box(
                body,
                xform=wp.transform(
                    wp.vec3(ROD_RADIUS + STRIPE_HALF_THICKNESS, 0.0, 0.0),
                    wp.quat_identity(),
                ),
                hx=STRIPE_HALF_THICKNESS,
                hy=STRIPE_HALF_WIDTH,
                hz=0.42 * float(segment_length),
                cfg=visual_cfg,
                color=STRIPE_COLOR,
                label=f"material_stripe_{index}",
            )

        for support, body in enumerate(self.anchor_bodies):
            outward = -1.0 if support == 0 else 1.0
            half_length = 0.5 * float(segment_lengths[0 if support == 0 else -1])
            builder.add_shape_box(
                body,
                xform=wp.transform(
                    wp.vec3(
                        0.0,
                        0.0,
                        outward * (half_length + SUPPORT_HALF_EXTENT),
                    ),
                    wp.quat_identity(),
                ),
                hx=SUPPORT_HALF_EXTENT,
                hy=SUPPORT_HALF_EXTENT,
                hz=SUPPORT_HALF_EXTENT,
                cfg=visual_cfg,
                color=SUPPORT_COLORS[support],
                label=f"rotating_support_{support}",
            )

        self.initial_anchor_poses = np.asarray(
            [builder.body_q[body] for body in self.anchor_bodies],
            dtype=np.float32,
        )
        self.kinematic_bodies = wp.array(
            self.anchor_bodies,
            dtype=wp.int32,
            device=self.device,
        )
        self.twist_rates = wp.array(
            (SUPPORT_TWIST_RATE, 0.0),
            dtype=wp.float32,
            device=self.device,
        )
        self.approach_rates = wp.array(
            (SUPPORT_APPROACH_RATE, 0.0),
            dtype=wp.float32,
            device=self.device,
        )
        self.target_x = wp.array(
            (
                self.initial_anchor_poses[0, 0] + SUPPORT_APPROACH_RATE * DRIVE_DURATION,
                self.initial_anchor_poses[1, 0],
            ),
            dtype=wp.float32,
            device=self.device,
        )
        return [default_capsule_half_extents(ROD_RADIUS, 0.5 * float(length)) for length in segment_lengths]

    def prepare_collision_update(self, dt: float) -> None:
        """Drive the supports at the 120 Hz collision cadence."""
        wp.launch(
            _rotate_supports_kernel,
            dim=2,
            inputs=(
                self.kinematic_bodies,
                self.twist_rates,
                self.approach_rates,
                self.target_x,
                dt,
                self.state.body_q,
                self.state.body_qd,
            ),
            device=self.device,
        )

    def configure_camera(self, viewer) -> None:
        """Frame the complete hanging span and both rotating cubes."""
        viewer.set_camera(
            pos=wp.vec3(0.0, -8.2, 1.70),
            pitch=-8.0,
            yaw=90.0,
        )

    def test_final(self) -> None:
        """Verify driven anchors, bounded motion, and stable attachment."""
        super().test_final()
        body_q = self.state.body_q.numpy()
        rod_q = body_q[self.rod_bodies]
        anchor_q = body_q[list(self.anchor_bodies)]

        drive_time = min(self.sim_time, DRIVE_DURATION)
        expected_left_x = self.initial_anchor_poses[0, 0] + SUPPORT_APPROACH_RATE * drive_time
        np.testing.assert_allclose(anchor_q[0, 0], expected_left_x, rtol=0.0, atol=2.0e-4)
        np.testing.assert_allclose(anchor_q[0, 1:3], self.initial_anchor_poses[0, 1:3], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(anchor_q[1], self.initial_anchor_poses[1], rtol=0.0, atol=1.0e-5)
        if self.sim_time > DRIVE_DURATION:
            orientation_dot = float(abs(np.dot(anchor_q[0, 3:7], self.initial_anchor_poses[0, 3:7])))
            expected_dot = abs(math.cos(0.5 * SUPPORT_TWIST_RATE * self.sim_time))
            np.testing.assert_allclose(orientation_dot, expected_dot, rtol=0.0, atol=2.0e-4)
        if float(np.max(np.abs(rod_q[:, :3]))) > 4.0:
            raise AssertionError("twisted rod left the expected scene bounds")
        if float(np.min(rod_q[:, 2])) < -1.5:
            raise AssertionError("twisted rod snapped or fell below the expected hanging span")


def _configure_parser(parser) -> None:
    parser.set_defaults(viewer="optix", num_frames=1230)


if __name__ == "__main__":
    run_ported_example(Example, _configure_parser)
