# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Settling metric based on the motion of committed body poses."""

import warp as wp

from newton._src.solvers.phoenx.body import MOTION_ARTICULATED, MOTION_DYNAMIC, MOTION_KINEMATIC, BodyContainer


@wp.struct
class MotionSleepHistory:
    previous_pose: wp.array[wp.transform]
    translation: wp.array[wp.vec3]
    rotation: wp.array[wp.vec3]
    remaining: wp.array[wp.float32]
    initialized: wp.array[wp.int32]


def create_motion_sleep_history(body_count: int, device) -> MotionSleepHistory:
    """Allocate persistent graph-safe history for the opt-in settling metric."""
    history = MotionSleepHistory()
    history.previous_pose = wp.zeros(body_count, dtype=wp.transform, device=device)
    history.translation = wp.zeros(body_count, dtype=wp.vec3, device=device)
    history.rotation = wp.zeros(body_count, dtype=wp.vec3, device=device)
    history.remaining = wp.full(body_count, 0.4, dtype=wp.float32, device=device)
    history.initialized = wp.zeros(body_count, dtype=wp.int32, device=device)
    return history


@wp.kernel(enable_backward=False)
def score_island_motion(
    bodies: BodyContainer,
    body_aabb_diagonal: wp.array[wp.float32],
    island_of_body: wp.array[wp.int32],
    step_dt: wp.float32,
    velocity_threshold: wp.float32,
    motion_threshold_squared: wp.float32,
    history: MotionSleepHistory,
    island_score: wp.array[wp.float32],
):
    """Mark islands that still move over the settling window or receive external input."""
    body = wp.tid()
    island = island_of_body[body]
    center = bodies.position[body]
    orientation = bodies.orientation[body]
    previous = history.previous_pose[body]
    history.previous_pose[body] = wp.transform(center, orientation)
    if history.initialized[body] == 0:
        history.initialized[body] = 1
        if island >= 0:
            wp.atomic_max(island_score, island, velocity_threshold)
        return

    if bodies.island_root[body] >= 0:
        history.remaining[body] = 0.4
        history.translation[body] = wp.vec3(0.0)
        history.rotation[body] = wp.vec3(0.0)
        return
    if island < 0:
        return

    motion = bodies.motion_type[body]
    radius = 0.5 * body_aabb_diagonal[body]
    if motion == MOTION_KINEMATIC or motion == MOTION_ARTICULATED:
        # Prescribed movers: the reduced articulation moves articulated links.
        speed = wp.length(bodies.velocity[body]) + radius * wp.length(bodies.angular_velocity[body])
        if speed >= velocity_threshold:
            wp.atomic_max(island_score, island, speed)
        return

    inverse_mass = bodies.inverse_mass[body]
    if motion != MOTION_DYNAMIC or inverse_mass <= 0.0:
        return

    remaining = history.remaining[body]
    reset = False
    if remaining < 0.2 or remaining < step_dt:
        delta_translation = center - wp.transform_get_translation(previous)
        delta_rotation = wp.quat_inverse(wp.transform_get_rotation(previous)) * orientation
        if delta_rotation[3] < 0.0:
            delta_rotation = -delta_rotation
        axis, angle = wp.quat_to_axis_angle(delta_rotation)
        translation = history.translation[body] + delta_translation
        rotation = history.rotation[body] + axis * angle

        # Inertia per unit mass converts angular motion to the equivalent
        # RMS material displacement for the body's mass distribution.
        inverse_inertia = bodies.inverse_inertia[body]
        angular_displacement_squared = wp.length_sq(rotation) * radius * radius
        if wp.determinant(inverse_inertia) > 0.0:
            angular_displacement_squared = wp.dot(rotation, wp.inverse(inverse_inertia) * rotation) * inverse_mass
        displacement_squared = wp.dot(translation, translation) + angular_displacement_squared
        reset = displacement_squared >= motion_threshold_squared
        history.translation[body] = translation
        history.rotation[body] = rotation
        if reset:
            remaining = 0.2 * wp.min(displacement_squared / motion_threshold_squared, 2.0)
            history.translation[body] = wp.vec3(0.0)
            history.rotation[body] = wp.vec3(0.0)

    if not reset:
        remaining = wp.max(0.0, remaining - step_dt)
    history.remaining[body] = remaining
    if remaining > 0.0 or wp.length_sq(bodies.force[body]) + wp.length_sq(bodies.torque[body]) > 0.0:
        wp.atomic_max(island_score, island, velocity_threshold)
