# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX Wilberforce pendulum
#
# A steel helical spring carries a rectangular mass. Axial oscillation of
# the mass unwinds the helix and excites torsional oscillation, reproducing
# the coupled motion shown in the Discrete Elastic Rods demo video.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_wilberforce_pendulum
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

# Material and discretization values reported for the Wilberforce pendulum
# in the reference video.
NUM_SEGMENTS = 100
YOUNGS_MODULUS = 209.0e9
SHEAR_MODULUS = 79.0e9
STEEL_DENSITY = 7850.0

NUM_TURNS = 5
SPRING_RADIUS = 0.065
WIRE_RADIUS = 0.0025
TURN_PITCH = 0.10
TOP_HEIGHT = 1.20
END_TAPER_TURNS = 1.0

MASS_HALF_EXTENTS = (0.035, 0.025, 0.080)
MASS_POLAR_INERTIA_SCALE = 7.85
INITIAL_VERTICAL_SPEED = -0.45

SPRING_COLOR = (0.66, 0.68, 0.70)
MASS_COLOR = (0.26, 0.27, 0.28)
ANCHOR_COLOR = (0.92, 0.92, 0.90)


def _relative_vertical_twist(pose_quaternion: np.ndarray, reference_quaternion: np.ndarray) -> float:
    """Return the magnitude of relative twist about the world vertical."""
    x1, y1, z1, w1 = (float(value) for value in pose_quaternion)
    x2, y2, z2, w2 = (float(value) for value in reference_quaternion)
    # pose * conjugate(reference), using the xyzw quaternion layout.
    relative_z = -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2
    relative_w = x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2
    twist_norm = math.hypot(relative_z, relative_w)
    if twist_norm == 0.0:
        return 0.0
    return 2.0 * math.atan2(abs(relative_z) / twist_norm, abs(relative_w) / twist_norm)


def _spring_points() -> np.ndarray:
    """Return a helix whose two attachments lie on its centerline."""
    points = np.empty((NUM_SEGMENTS + 1, 3), dtype=np.float32)
    for index in range(NUM_SEGMENTS + 1):
        turns = NUM_TURNS * index / NUM_SEGMENTS
        angle = 2.0 * math.pi * turns

        # A cubic ramp over the first and last turn brings both attachment
        # points onto the central axis with zero radial slope. Besides matching
        # the reference spring, this avoids an eccentric top clamp and mass.
        taper_coordinate = min(turns, NUM_TURNS - turns) / END_TAPER_TURNS
        taper_coordinate = min(max(taper_coordinate, 0.0), 1.0)
        radius_scale = taper_coordinate * taper_coordinate * (3.0 - 2.0 * taper_coordinate)
        radius = SPRING_RADIUS * radius_scale
        points[index] = (
            radius * math.cos(angle),
            radius * math.sin(angle),
            TOP_HEIGHT - TURN_PITCH * turns,
        )
    return points


class Example(PortedExample):
    """Demonstrate coupled axial and torsional spring oscillation."""

    fps = 60
    sim_substeps = 15
    solver_iterations = 8
    velocity_iterations = 1
    broad_phase = "sap"
    step_layout = "single_world"
    mass_splitting = True
    mass_splitting_color_group_size = 3
    max_colored_partitions = 8
    shape_pairs_max = 8192
    show_contacts = True
    evaluate_fk = False
    step_report_label = "WilberforcePendulum"

    def build_scene(self, builder: newton.ModelBuilder):
        """Build the steel helix, fixed top anchor, and hanging mass."""
        points = _spring_points()
        rod = newton.Rod(
            points,
            radius=WIRE_RADIUS,
            youngs_modulus=YOUNGS_MODULUS,
            shear_modulus=SHEAR_MODULUS,
        )
        rod_cfg = newton.ModelBuilder.ShapeConfig(
            density=STEEL_DENSITY,
            mu=0.4,
            restitution=0.0,
            gap=0.0005,
        )
        bodies, _ = builder.add_rod(
            rod=rod,
            cfg=rod_cfg,
            stretch_damping=2.0,
            shear_damping=2.0,
            bend_damping=0.01,
            twist_damping=0.01,
            label="wilberforce_spring",
            color=SPRING_COLOR,
            body_frame_origin="com",
        )
        self.spring_bodies = [int(body) for body in bodies]
        self.anchor_body = self.spring_bodies[0]
        self.mass_body = self.spring_bodies[-1]

        # Clamp the first spring segment. A non-colliding sphere makes the
        # suspension point visible without introducing an artificial contact.
        builder.body_mass[self.anchor_body] = 0.0
        builder.body_inv_mass[self.anchor_body] = 0.0
        builder.body_inertia[self.anchor_body] = wp.mat33(0.0)
        builder.body_inv_inertia[self.anchor_body] = wp.mat33(0.0)
        visual_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
        anchor_world = wp.transform(wp.vec3(*points[0]), wp.quat_identity())
        anchor_local = wp.transform_inverse(builder.body_q[self.anchor_body]) * anchor_world
        builder.add_shape_sphere(
            self.anchor_body,
            xform=anchor_local,
            radius=0.025,
            cfg=visual_cfg,
            color=ANCHOR_COLOR,
            label="top_anchor",
        )

        # Attach the dense block directly to the final spring body. Keeping it
        # in the same rigid body avoids a redundant fixed constraint while its
        # off-center inertia is still accumulated by ModelBuilder.
        mass_center = points[-1].copy()
        mass_center[2] -= MASS_HALF_EXTENTS[2] + WIRE_RADIUS
        mass_world = wp.transform(wp.vec3(*mass_center), wp.quat_identity())
        mass_local = wp.transform_inverse(builder.body_q[self.mass_body]) * mass_world
        mass_cfg = newton.ModelBuilder.ShapeConfig(
            density=STEEL_DENSITY,
            mu=0.4,
            restitution=0.0,
            gap=0.001,
        )
        builder.add_shape_box(
            self.mass_body,
            xform=mass_local,
            hx=MASS_HALF_EXTENTS[0],
            hy=MASS_HALF_EXTENTS[1],
            hz=MASS_HALF_EXTENTS[2],
            cfg=mass_cfg,
            color=MASS_COLOR,
            label="pendulum_mass",
        )

        # A slim pointer on the top face makes the coupled torsional mode
        # readable without changing the block mass or collision geometry.
        marker_center = mass_center.copy()
        marker_center[0] += 0.018
        marker_center[2] += MASS_HALF_EXTENTS[2] + 0.0015
        marker_world = wp.transform(wp.vec3(*marker_center), wp.quat_identity())
        marker_local = wp.transform_inverse(builder.body_q[self.mass_body]) * marker_world
        builder.add_shape_box(
            self.mass_body,
            xform=marker_local,
            hx=0.018,
            hy=0.004,
            hz=0.0015,
            cfg=visual_cfg,
            color=ANCHOR_COLOR,
            label="torsion_pointer",
        )

        # Wilberforce pendulums tune an internal flywheel or adjustable masses
        # until the axial and torsional modes coincide. Keep the visible mass
        # and its translational inertia unchanged, while representing that
        # internal polar inertia in the assembled lower body.
        mass_inertia = np.asarray(builder.body_inertia[self.mass_body], dtype=np.float64).reshape(3, 3)
        flywheel_polar_inertia = mass_inertia[2, 2] * (MASS_POLAR_INERTIA_SCALE - 1.0)
        mass_inertia[0, 0] += 0.5 * flywheel_polar_inertia
        mass_inertia[1, 1] += 0.5 * flywheel_polar_inertia
        mass_inertia[2, 2] += flywheel_polar_inertia
        builder.body_inertia[self.mass_body] = wp.mat33(mass_inertia.astype(np.float32))
        builder.body_inv_inertia[self.mass_body] = wp.inverse(builder.body_inertia[self.mass_body])

        # Seed a smooth axial mode with zero initial angular velocity. Any
        # subsequent rotation is therefore generated by helix coupling.
        for index, body in enumerate(self.spring_bodies):
            speed = INITIAL_VERTICAL_SPEED * index / (NUM_SEGMENTS - 1)
            builder.body_qd[body] = wp.spatial_vector(0.0, 0.0, speed, 0.0, 0.0, 0.0)

        self.initial_anchor_pose = np.asarray(builder.body_q[self.anchor_body], dtype=np.float32)
        self.initial_mass_pose = np.asarray(builder.body_q[self.mass_body], dtype=np.float32)
        self._test_min_mass_z = math.inf
        self._test_max_mass_z = -math.inf
        self._test_max_rotation = 0.0

        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return [default_capsule_half_extents(WIRE_RADIUS, 0.5 * float(length)) for length in segment_lengths]

    def configure_camera(self, viewer) -> None:
        """Frame the complete hanging spring and mass."""
        viewer.set_camera(
            pos=wp.vec3(1.00, -1.55, 0.90),
            pitch=-2.0,
            yaw=145.0,
        )

    def test_post_step(self) -> None:
        """Record axial and angular excursion during headless tests."""
        pose = self.state.body_q.numpy()[self.mass_body]
        self._test_min_mass_z = min(self._test_min_mass_z, float(pose[2]))
        self._test_max_mass_z = max(self._test_max_mass_z, float(pose[2]))
        twist = _relative_vertical_twist(pose[3:7], self.initial_mass_pose[3:7])
        self._test_max_rotation = max(self._test_max_rotation, twist)

    def test_final(self) -> None:
        """Verify stable axial motion produces coupled torsional motion."""
        super().test_final()
        anchor_pose = self.state.body_q.numpy()[self.anchor_body]
        np.testing.assert_allclose(anchor_pose, self.initial_anchor_pose, rtol=0.0, atol=1.0e-6)
        axial_excursion = self._test_max_mass_z - self._test_min_mass_z
        if axial_excursion < 0.025:
            raise AssertionError(f"pendulum axial excursion is too small ({axial_excursion:.4f} m)")
        if self._test_max_rotation < math.radians(45.0):
            raise AssertionError(
                "vertical excitation did not produce torsional motion "
                f"(maximum rotation={math.degrees(self._test_max_rotation):.2f} deg)"
            )
        mass_position = self.state.body_q.numpy()[self.mass_body, :3]
        if np.linalg.norm(mass_position[:2]) > 0.35 or not 0.20 < mass_position[2] < 1.10:
            raise AssertionError(f"pendulum left the expected scene bounds ({mass_position})")


def _configure_parser(parser) -> None:
    parser.set_defaults(viewer="optix", num_frames=600)


if __name__ == "__main__":
    run_ported_example(Example, _configure_parser)
