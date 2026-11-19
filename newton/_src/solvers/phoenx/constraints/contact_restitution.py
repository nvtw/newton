# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid impact velocity and pose transport at a swept time of contact."""

import warp as wp

from newton._src.solvers.phoenx.adapter_kernels import _contact_impulse
from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, BodyContainer, mat33_from_sym6, sym6_from_mat33
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactViews
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.helpers.math_helpers import rotate_inertia
from newton._src.solvers.phoenx.materials import MaterialData, resolve_restitution_in_kernel
from newton._src.solvers.phoenx.simulation_kernels import _rotation_quaternion


@wp.kernel(enable_backward=False)
def prepare_restitution_targets(
    contacts: ContactViews,
    bodies: BodyContainer,
    materials: wp.array[MaterialData],
    shape_material: wp.array[wp.int32],
    gravity: wp.array[wp.vec3f],
    dt: wp.float32,
    detection_interval: wp.float32,
    targets: wp.array[wp.vec2f],
):
    k = wp.tid()
    if k >= contacts.rigid_contact_count[0]:
        return
    targets[k] = wp.vec2f(0.0)
    s0 = contacts.rigid_contact_shape0[k]
    s1 = contacts.rigid_contact_shape1[k]
    if s0 < 0 or s1 < 0 or s0 >= shape_material.shape[0] or s1 >= shape_material.shape[0]:
        return
    restitution = resolve_restitution_in_kernel(materials, shape_material[s0], shape_material[s1])
    if restitution == 0.0:
        return

    b0 = contacts.shape_body[s0]
    b1 = contacts.shape_body[s1]
    n = contacts.rigid_contact_normal[k]
    r0 = wp.quat_rotate(bodies.orientation[b0], contacts.rigid_contact_point0[k] - bodies.body_com[b0])
    r1 = wp.quat_rotate(bodies.orientation[b1], contacts.rigid_contact_point1[k] - bodies.body_com[b1])
    v0 = bodies.velocity[b0] + wp.cross(bodies.angular_velocity[b0], r0)
    v1 = bodies.velocity[b1] + wp.cross(bodies.angular_velocity[b1], r1)
    if contacts.rigid_contact_surface_velocity.shape[0] > 0:
        v1 += contacts.rigid_contact_surface_velocity[k]
    gap = wp.dot(bodies.position[b1] - bodies.position[b0] + r1 - r0, n)
    gap -= contacts.rigid_contact_margin0[k] + contacts.rigid_contact_margin1[k]

    world = wp.max(wp.max(bodies.world_id[b0], bodies.world_id[b1]), 0)
    normal_gravity = wp.dot(gravity[world], n)
    acceleration = wp.float32(0.0)
    if bodies.inverse_mass[b1] > 0.0 and bodies.affected_by_gravity[b1] != 0:
        acceleration += normal_gravity
    if bodies.inverse_mass[b0] > 0.0 and bodies.affected_by_gravity[b0] != 0:
        acceleration -= normal_gravity
    # The force stage already advanced velocities to the end of the substep.
    closing = -wp.dot(v1 - v0, n) + acceleration * dt
    if closing <= 2.0 * wp.abs(acceleration) * dt:
        return

    # Solve gap - closing*t + acceleration*t²/2 = 0. A newly observed
    # penetrating contact can have crossed during the previous collision interval.
    discriminant = closing * closing - 2.0 * acceleration * gap
    if discriminant >= 0.0:
        arrival = 2.0 * gap / (closing + wp.sqrt(discriminant))
        if arrival <= dt:
            arrival = wp.max(-detection_interval, arrival)
            outgoing = restitution * (closing - acceleration * arrival) + acceleration * (dt - arrival)
            targets[k] = wp.vec2f(outgoing, arrival)
            return
    # A distant speculative row permits uninterrupted approach up to contact.
    targets[k] = wp.vec2f(-gap / dt, 0.0)


@wp.kernel(enable_backward=False)
def accumulate_impact_pose_corrections(
    contacts: ContactViews,
    bodies: BodyContainer,
    points: ContactContainer,
    point_constraint: wp.array[wp.int32],
    corrections: wp.array[wp.spatial_vector],
):
    k = wp.tid()
    if k >= contacts.rigid_contact_count[0] or point_constraint[k] < 0:
        return
    arrival = contacts.restitution_target[k][1]
    if arrival == 0.0:
        return
    # The ordinary integrator transports the solved impulse over a full
    # substep. Subtract its contribution over the time preceding impact.
    impulse_time = _contact_impulse(points, k) * arrival
    for side in range(2):
        body = contacts.shape_body[contacts.rigid_contact_shape0[k]]
        point = contacts.rigid_contact_point0[k]
        correction = impulse_time
        if side == 1:
            body = contacts.shape_body[contacts.rigid_contact_shape1[k]]
            point = contacts.rigid_contact_point1[k]
            correction = -impulse_time
        if bodies.motion_type[body] == MOTION_DYNAMIC and bodies.island_root[body] < 0:
            arm = wp.quat_rotate(bodies.orientation[body], point - bodies.body_com[body])
            linear = bodies.inverse_mass[body] * correction
            angular = mat33_from_sym6(bodies.inverse_inertia_world[body]) * wp.cross(arm, correction)
            wp.atomic_add(corrections, body, wp.spatial_vector(linear, angular))


@wp.kernel(enable_backward=False)
def apply_impact_pose_corrections(bodies: BodyContainer, corrections: wp.array[wp.spatial_vector]):
    body = wp.tid()
    delta = corrections[body]
    linear = wp.spatial_top(delta)
    angular = wp.spatial_bottom(delta)
    if wp.length_sq(linear) == 0.0 and wp.length_sq(angular) == 0.0:
        return
    corrections[body] = wp.spatial_vector()
    bodies.position[body] += linear
    if wp.length_sq(angular) > 0.0:
        momentum = wp.inverse(mat33_from_sym6(bodies.inverse_inertia_world[body])) * bodies.angular_velocity[body]
        orientation = wp.normalize(_rotation_quaternion(angular, 1.0) * bodies.orientation[body])
        inverse_inertia = rotate_inertia(wp.quat_to_matrix(orientation), bodies.inverse_inertia[body])
        bodies.orientation[body] = orientation
        bodies.inverse_inertia_world[body] = sym6_from_mat33(inverse_inertia)
        bodies.angular_velocity[body] = inverse_inertia * momentum
