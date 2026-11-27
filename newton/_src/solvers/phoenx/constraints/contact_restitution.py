# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid impact velocity and pose transport at a swept time of contact."""

import warp as wp

from newton._src.solvers.phoenx.adapter_kernels import _contact_impulse
from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, BodyContainer, mat33_from_sym6, sym6_from_mat33
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactViews
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer, cc_get_normal_lambda
from newton._src.solvers.phoenx.helpers.math_helpers import rotate_inertia
from newton._src.solvers.phoenx.materials import MaterialData, resolve_restitution_in_kernel
from newton._src.solvers.phoenx.simulation_kernels import _rotation_quaternion


@wp.func
def impact_sweep(arrival: float, dt: float, count: int):
    """Assign a boundary impact to the sweep ending at that boundary."""
    return wp.clamp(int(wp.ceil(arrival * float(count) / dt)) - 1, 0, count - 1)


@wp.func
def _normal_point_acceleration(
    bodies: BodyContainer, body: int, arm: wp.vec3f, normal: wp.vec3f, gravity: wp.array[wp.vec3f]
):
    if bodies.motion_type[body] != MOTION_DYNAMIC or bodies.island_root[body] >= 0:
        return 0.0
    linear = bodies.inverse_mass[body] * bodies.force[body]
    if bodies.affected_by_gravity[body] != 0:
        linear += gravity[bodies.world_id[body]]
    angular = mat33_from_sym6(bodies.inverse_inertia_world[body]) * bodies.torque[body]
    return wp.dot(linear + wp.cross(angular, arm), normal)


@wp.kernel(enable_backward=False)
def prepare_restitution_targets(
    contacts: ContactViews,
    bodies: BodyContainer,
    materials: wp.array[MaterialData],
    shape_material: wp.array[wp.int32],
    gravity: wp.array[wp.vec3f],
    dt: wp.float32,
    detection_interval: wp.float32,
    sweeps: int,
    point_constraint: wp.array[int],
    owner: wp.array[int],
    offset: int,
    jointed: wp.array[int],
    points: ContactContainer,
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

    # A touching witness carrying compressive load belongs to a continuing contact.
    if gap <= 0.0 and cc_get_normal_lambda(points, k) > 0.0:
        return

    # Undo only the force impulse that each endpoint has already received.
    a0 = _normal_point_acceleration(bodies, b0, r0, n, gravity)
    a1 = _normal_point_acceleration(bodies, b1, r1, n, gravity)
    acceleration = a1 - a0
    per_sweep = float(0.0)
    if sweeps > 1:
        if jointed[b0] == 0:
            per_sweep -= a0
        if jointed[b1] == 0:
            per_sweep += a1
    closing = -wp.dot(v1 - v0, n) + acceleration * dt - per_sweep * (dt - dt / float(sweeps))
    if closing <= 2.0 * wp.abs(acceleration) * dt:
        return

    # Solve gap - closing*t + acceleration*t²/2 = 0. A newly observed
    # penetrating contact can have crossed during the previous collision interval.
    discriminant = closing * closing - 2.0 * acceleration * gap
    if discriminant >= 0.0:
        arrival = 2.0 * gap / (closing + wp.sqrt(discriminant))
        if arrival <= dt:
            if arrival < -detection_interval:
                return
            outgoing = restitution * (closing - acceleration * arrival) + acceleration * (dt - arrival)
            if sweeps > 1:
                column = point_constraint[k] - offset
                if column >= 0 and owner[column] < 0:
                    # The regular contact solves at the end of its impact sweep.
                    # Subsequent sweeps will apply the remaining external force.
                    end = dt * float(impact_sweep(arrival, dt, sweeps) + 1) / float(sweeps)
                    outgoing -= per_sweep * (dt - end)
            targets[k] = wp.vec2f(outgoing, arrival)
            return
    # A distant speculative row permits uninterrupted approach up to contact.
    targets[k] = wp.vec2f(-gap / dt, 0.0)


#: Fixed-point scale of the summed pose corrections [1/m, 1/rad]: 2^-40 resolution, 8e6 range.
_CORRECTION_SCALE = wp.constant(wp.float64(2.0**40))


@wp.func
def _add_fixed_point(corrections: wp.array2d[wp.int64], body: int, first: int, value: wp.vec3f):
    """Accumulate ``value`` in fixed point, so the sum is independent of the arrival order."""
    for axis in range(3):
        wp.atomic_add(corrections, body, first + axis, wp.int64(wp.round(wp.float64(value[axis]) * _CORRECTION_SCALE)))


@wp.func
def _fixed_point_vec3(corrections: wp.array2d[wp.int64], body: int, first: int) -> wp.vec3f:
    return wp.vec3f(
        wp.float32(wp.float64(corrections[body, first]) / _CORRECTION_SCALE),
        wp.float32(wp.float64(corrections[body, first + 1]) / _CORRECTION_SCALE),
        wp.float32(wp.float64(corrections[body, first + 2]) / _CORRECTION_SCALE),
    )


@wp.func
def accumulate_impact_delay(
    contacts: ContactViews,
    bodies: BodyContainer,
    points: ContactContainer,
    corrections: wp.array2d[wp.int64],
    k: int,
    arrival: float,
):
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
            _add_fixed_point(corrections, body, 0, linear)
            _add_fixed_point(corrections, body, 3, angular)


@wp.kernel(enable_backward=False)
def accumulate_impact_pose_corrections(
    contacts: ContactViews,
    bodies: BodyContainer,
    points: ContactContainer,
    point_constraint: wp.array[wp.int32],
    corrections: wp.array2d[wp.int64],
    owner: wp.array[int],
    offset: int,
    temporal: bool,
    sweeps: int,
    dt: float,
):
    k = wp.tid()
    if k >= contacts.rigid_contact_count[0] or point_constraint[k] < 0:
        return
    arrival = contacts.restitution_target[k][1]
    if arrival == 0.0:
        return
    if temporal:
        column = point_constraint[k] - offset
        if column >= 0 and owner[column] < 0:
            # Earlier sweeps already accounted for their impulses at gap refresh.
            if impact_sweep(arrival, dt, sweeps) < sweeps - 1:
                return
            arrival -= dt * float(sweeps - 1) / float(sweeps)
    accumulate_impact_delay(contacts, bodies, points, corrections, k, arrival)


@wp.kernel(enable_backward=False)
def apply_impact_pose_corrections(bodies: BodyContainer, corrections: wp.array2d[wp.int64]):
    body = wp.tid()
    linear = _fixed_point_vec3(corrections, body, 0)
    angular = _fixed_point_vec3(corrections, body, 3)
    if wp.length_sq(linear) == 0.0 and wp.length_sq(angular) == 0.0:
        return
    for axis in range(6):
        corrections[body, axis] = wp.int64(0)
    bodies.position[body] += linear
    if wp.length_sq(angular) > 0.0:
        momentum = wp.inverse(mat33_from_sym6(bodies.inverse_inertia_world[body])) * bodies.angular_velocity[body]
        orientation = wp.normalize(_rotation_quaternion(angular, 1.0) * bodies.orientation[body])
        inverse_inertia = rotate_inertia(wp.quat_to_matrix(orientation), bodies.inverse_inertia[body])
        bodies.orientation[body] = orientation
        bodies.inverse_inertia_world[body] = sym6_from_mat33(inverse_inertia)
        bodies.angular_velocity[body] = inverse_inertia * momentum
