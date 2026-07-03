# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lean rigid-contact rows backed by a reduced articulation mass operator."""

from __future__ import annotations

import warp as wp

from newton._src.geometry.types import GeoType
from newton._src.solvers.phoenx.body import MOTION_ARTICULATED, BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_get_friction,
    contact_get_friction_dynamic,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_bias,
    cc_get_bias_t1,
    cc_get_bias_t2,
    cc_get_eff_n,
    cc_get_eff_t1,
    cc_get_eff_t2,
    cc_get_local_p0,
    cc_get_local_p1,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_start_gap,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_bias,
    cc_set_bias_t1,
    cc_set_bias_t2,
    cc_set_eff_n,
    cc_set_eff_t1,
    cc_set_eff_t2,
    cc_set_r0,
    cc_set_r1,
)
from newton._src.solvers.phoenx.constraints.contact_endpoint import _articulation_pair_response
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_velocity_update_no_soft_pd
from newton._src.solvers.phoenx.helpers.math_helpers import apply_body_spatial_impulse

_vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def _point_velocity(bodies: BodyContainer, body: wp.int32, point: wp.vec3f) -> wp.vec3f:
    return bodies.velocity[body] + wp.cross(bodies.angular_velocity[body], point - bodies.position[body])


@wp.func
def _rigid_endpoint_inverse_mass(
    bodies: BodyContainer,
    body: wp.int32,
    point: wp.vec3f,
    direction: wp.vec3f,
) -> wp.float32:
    inverse_mass = bodies.inverse_mass[body]
    if inverse_mass == wp.float32(0.0):
        return wp.float32(0.0)
    angular_jacobian = wp.cross(point - bodies.position[body], direction)
    inverse_inertia = mat33_from_sym6(bodies.inverse_inertia_world[body])
    return inverse_mass + wp.dot(angular_jacobian, inverse_inertia * angular_jacobian)


@wp.func
def _apply_rigid_endpoint_impulse(
    bodies: BodyContainer,
    body: wp.int32,
    point: wp.vec3f,
    impulse: wp.vec3f,
):
    inverse_mass = bodies.inverse_mass[body]
    if inverse_mass == wp.float32(0.0):
        return
    velocity, angular_velocity = apply_body_spatial_impulse(
        bodies.velocity[body],
        bodies.angular_velocity[body],
        inverse_mass,
        mat33_from_sym6(bodies.inverse_inertia_world[body]),
        impulse,
        wp.cross(point - bodies.position[body], impulse),
    )
    bodies.velocity[body] = velocity
    bodies.angular_velocity[body] = angular_velocity


@wp.func
def _pair_inverse_mass(
    bodies: BodyContainer,
    body0: wp.int32,
    point0: wp.vec3f,
    body1: wp.int32,
    point1: wp.vec3f,
    direction: wp.vec3f,
) -> wp.float32:
    articulation0 = bodies.reduced.body_articulation[body0]
    articulation1 = bodies.reduced.body_articulation[body1]
    result = wp.float32(0.0)

    for side in range(2):
        body = body0
        point = point0
        if side == 1:
            body = body1
            point = point1
        if bodies.motion_type[body] != MOTION_ARTICULATED:
            result += _rigid_endpoint_inverse_mass(bodies, body, point, direction)

    for group in range(2):
        articulation = articulation0
        if group == 1:
            articulation = articulation1
        if articulation < wp.int32(0) or (group == 1 and articulation == articulation0):
            continue

        slot0 = wp.int32(-1)
        slot1 = wp.int32(-1)
        group_point0 = wp.vec3f(0.0)
        group_point1 = wp.vec3f(0.0)
        impulse0 = wp.vec3f(0.0)
        impulse1 = wp.vec3f(0.0)
        if articulation0 == articulation:
            slot0 = body0
            group_point0 = point0
            impulse0 = -direction
        if articulation1 == articulation:
            if slot0 < wp.int32(0):
                slot0 = body1
                group_point0 = point1
                impulse0 = direction
            else:
                slot1 = body1
                group_point1 = point1
                impulse1 = direction
        result += _articulation_pair_response(
            bodies,
            slot0,
            group_point0,
            impulse0,
            slot1,
            group_point1,
            impulse1,
            wp.bool(False),
        )
    return result


@wp.func
def _apply_pair_impulse(
    bodies: BodyContainer,
    body0: wp.int32,
    point0: wp.vec3f,
    body1: wp.int32,
    point1: wp.vec3f,
    impulse_on_body1: wp.vec3f,
):
    articulation0 = bodies.reduced.body_articulation[body0]
    articulation1 = bodies.reduced.body_articulation[body1]

    for side in range(2):
        body = body0
        point = point0
        impulse = -impulse_on_body1
        if side == 1:
            body = body1
            point = point1
            impulse = impulse_on_body1
        if bodies.motion_type[body] != MOTION_ARTICULATED:
            _apply_rigid_endpoint_impulse(bodies, body, point, impulse)

    for group in range(2):
        articulation = articulation0
        if group == 1:
            articulation = articulation1
        if articulation < wp.int32(0) or (group == 1 and articulation == articulation0):
            continue

        slot0 = wp.int32(-1)
        slot1 = wp.int32(-1)
        group_point0 = wp.vec3f(0.0)
        group_point1 = wp.vec3f(0.0)
        impulse0 = wp.vec3f(0.0)
        impulse1 = wp.vec3f(0.0)
        if articulation0 == articulation:
            slot0 = body0
            group_point0 = point0
            impulse0 = -impulse_on_body1
        if articulation1 == articulation:
            if slot0 < wp.int32(0):
                slot0 = body1
                group_point0 = point1
                impulse0 = impulse_on_body1
            else:
                slot1 = body1
                group_point1 = point1
                impulse1 = impulse_on_body1
        _articulation_pair_response(
            bodies,
            slot0,
            group_point0,
            impulse0,
            slot1,
            group_point1,
            impulse1,
            wp.bool(True),
        )


@wp.func
def reduced_contact_deferred_owner(
    columns: ContactColumnContainer,
    column: wp.int32,
    bodies: BodyContainer,
) -> wp.int32:
    """Return the articulation for an articulation-immutable contact."""
    body0 = contact_get_body1(columns, column)
    body1 = contact_get_body2(columns, column)
    articulation0 = bodies.reduced.body_articulation[body0]
    articulation1 = bodies.reduced.body_articulation[body1]
    if articulation0 >= wp.int32(0) and articulation1 < wp.int32(0):
        if bodies.inverse_mass[body1] == wp.float32(0.0):
            return articulation0
    if articulation1 >= wp.int32(0) and articulation0 < wp.int32(0):
        if bodies.inverse_mass[body0] == wp.float32(0.0):
            return articulation1
    return wp.int32(-1)


@wp.func
def _reduced_point_local(bodies: BodyContainer, body: wp.int32, point: wp.vec3f) -> wp.vec3f:
    link = body - wp.int32(1)
    local_com = wp.transform_get_translation(bodies.reduced.body_q_com[link])
    return local_com + point - bodies.position[body]


@wp.func
def _response_matrix_multiply(
    bodies: BodyContainer,
    body: wp.int32,
    wrench: wp.spatial_vector,
) -> wp.spatial_vector:
    result = wp.spatial_vector()
    data = bodies.reduced
    for column in range(6):
        result += data.impulse_response[body, column] * wrench[column]
    return result


@wp.func
def _deferred_inverse_mass(
    bodies: BodyContainer,
    body0: wp.int32,
    point0: wp.vec3f,
    body1: wp.int32,
    point1: wp.vec3f,
    direction: wp.vec3f,
) -> wp.float32:
    body = body0
    point = point0
    impulse = -direction
    if bodies.reduced.body_articulation[body] < wp.int32(0):
        body = body1
        point = point1
        impulse = direction
    link = body - wp.int32(1)
    point_local = _reduced_point_local(bodies, body, point)
    wrench = wp.spatial_vector(impulse, wp.cross(point_local, impulse))
    return wp.dot(wrench, _response_matrix_multiply(bodies, link, wrench))


@wp.func
def _apply_deferred_impulse(
    bodies: BodyContainer,
    body0: wp.int32,
    point0: wp.vec3f,
    body1: wp.int32,
    point1: wp.vec3f,
    impulse_on_body1: wp.vec3f,
):
    body = body0
    point = point0
    impulse = -impulse_on_body1
    if bodies.reduced.body_articulation[body] < wp.int32(0):
        body = body1
        point = point1
        impulse = impulse_on_body1

    data = bodies.reduced
    link = body - wp.int32(1)
    point_local = _reduced_point_local(bodies, body, point)
    wrench = wp.spatial_vector(impulse, wp.cross(point_local, impulse))
    joint = data.body_joint[link]
    while joint >= wp.int32(0):
        data.deferred_wrench[link] = data.deferred_wrench[link] + wrench
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        projected = _vec6(0.0)
        reduced = _vec6(0.0)
        for row in range(6):
            if wp.int32(row) < dof_count:
                projected[row] = wp.dot(data.joint_s[dof_start + wp.int32(row)], wrench)
        for row in range(6):
            if wp.int32(row) < dof_count:
                for column in range(6):
                    if wp.int32(column) < dof_count:
                        reduced[row] += data.joint_d_inv[dof_start + wp.int32(row), column] * projected[column]
        propagated = wrench
        for row in range(6):
            if wp.int32(row) < dof_count:
                propagated -= data.joint_u[dof_start + wp.int32(row)] * reduced[row]
        link = data.joint_parent[joint]
        wrench = propagated
        if link >= wp.int32(0):
            joint = data.body_joint[link]
        else:
            joint = wp.int32(-1)


@wp.func
def _deferred_link_delta_twist(bodies: BodyContainer, body_slot: wp.int32) -> wp.spatial_vector:
    data = bodies.reduced
    body = body_slot - wp.int32(1)
    delta = wp.spatial_vector()
    start = data.body_path_start[body]
    end = data.body_path_start[body + wp.int32(1)]
    for path_index in range(start, end):
        joint = data.body_path_joint[path_index]
        child = data.joint_child[joint]
        wrench = data.deferred_wrench[child]
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        rhs = _vec6(0.0)
        response = _vec6(0.0)
        for row in range(6):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                rhs[row] = wp.dot(data.joint_s[dof], wrench) - wp.dot(data.joint_u[dof], delta)
        for row in range(6):
            if wp.int32(row) < dof_count:
                for column in range(6):
                    if wp.int32(column) < dof_count:
                        response[row] += data.joint_d_inv[dof_start + wp.int32(row), column] * rhs[column]
        for row in range(6):
            if wp.int32(row) < dof_count:
                delta += data.joint_s[dof_start + wp.int32(row)] * response[row]
    return delta


@wp.func
def _deferred_point_velocity(
    bodies: BodyContainer,
    body: wp.int32,
    point: wp.vec3f,
) -> wp.vec3f:
    velocity = _point_velocity(bodies, body, point)
    if bodies.reduced.body_articulation[body] >= wp.int32(0):
        delta = _deferred_link_delta_twist(bodies, body)
        velocity += wp.spatial_top(delta) + wp.cross(
            wp.spatial_bottom(delta), _reduced_point_local(bodies, body, point)
        )
    return velocity


@wp.func
def _uses_start_gap(contacts: ContactViews, contact: wp.int32) -> wp.bool:
    shape0 = contacts.rigid_contact_shape0[contact]
    shape1 = contacts.rigid_contact_shape1[contact]
    shape_count = contacts.shape_type.shape[0]
    result = wp.bool(False)
    if shape0 >= wp.int32(0) and shape0 < shape_count:
        shape_type = contacts.shape_type[shape0]
        result = (
            shape_type == wp.int32(GeoType.MESH)
            or shape_type == wp.int32(GeoType.HFIELD)
            or shape_type == wp.int32(GeoType.TETRAHEDRON)
        )
    if shape1 >= wp.int32(0) and shape1 < shape_count:
        shape_type = contacts.shape_type[shape1]
        result = result or (
            shape_type == wp.int32(GeoType.MESH)
            or shape_type == wp.int32(GeoType.HFIELD)
            or shape_type == wp.int32(GeoType.TETRAHEDRON)
        )
    return result


@wp.func
def reduced_contact_prepare(
    columns: ContactColumnContainer,
    column: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    contacts_state: ContactContainer,
    contacts: ContactViews,
    use_deferred: wp.bool,
    apply_warmstart: wp.bool,
    compute_effective_mass: wp.bool,
):
    """Prepare effective masses, TGS biases, and warm-start impulses.

    Pass ``compute_effective_mass=False`` from the block-owned contact
    gather: the packed row builder derives exact effective masses from the
    generalized response rows and overwrites the slots before any solve
    reads them, so the per-direction inverse-mass traversals here are dead
    work on that path.
    """
    body0 = contact_get_body1(columns, column)
    body1 = contact_get_body2(columns, column)
    first = contact_get_contact_first(columns, column)
    count = contact_get_contact_count(columns, column)
    dt = wp.float32(1.0) / idt
    bias_rate, _mass_coeff, _impulse_coeff = soft_constraint_coefficients(
        DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt
    )

    for offset in range(count):
        contact = first + offset
        normal = cc_get_normal(contacts_state, contact)
        tangent0 = cc_get_tangent1(contacts_state, contact)
        tangent1 = wp.cross(normal, tangent0)
        local0 = cc_get_local_p0(contacts_state, contact)
        local1 = cc_get_local_p1(contacts_state, contact)
        point0 = (
            bodies.position[body0]
            + wp.quat_rotate(bodies.orientation[body0], local0 - bodies.body_com[body0])
            + contacts.rigid_contact_margin0[contact] * normal
        )
        point1 = (
            bodies.position[body1]
            + wp.quat_rotate(bodies.orientation[body1], local1 - bodies.body_com[body1])
            - contacts.rigid_contact_margin1[contact] * normal
        )
        contact_point = wp.float32(0.5) * (point0 + point1)

        effective_mass = wp.vec3f(0.0)
        if compute_effective_mass:
            for row in range(3):
                direction = normal
                if row == 1:
                    direction = tangent0
                elif row == 2:
                    direction = tangent1
                inverse_mass = wp.float32(0.0)
                if use_deferred:
                    inverse_mass = _deferred_inverse_mass(bodies, body0, contact_point, body1, contact_point, direction)
                else:
                    inverse_mass = _pair_inverse_mass(bodies, body0, contact_point, body1, contact_point, direction)
                if inverse_mass > wp.float32(1.0e-12):
                    effective_mass[row] = wp.float32(1.0) / inverse_mass

        separation = point1 - point0
        gap = wp.dot(separation, normal)
        solver_gap = gap
        if _uses_start_gap(contacts, contact):
            start_gap = cc_get_start_gap(contacts_state, contact)
            if start_gap > wp.float32(0.0) and solver_gap < start_gap:
                solver_gap = start_gap
        bias = solver_gap * idt if solver_gap > wp.float32(0.0) else gap * bias_rate
        bias = wp.clamp(bias, wp.float32(-2.0), wp.float32(10.0))
        bias_t0 = wp.float32(0.0)
        bias_t1 = wp.float32(0.0)
        if solver_gap <= wp.float32(0.0) and gap <= wp.float32(0.002):
            bias_t0 = wp.float32(0.08) * wp.clamp(wp.dot(separation, tangent0), -0.001, 0.001) * idt
            bias_t1 = wp.float32(0.08) * wp.clamp(wp.dot(separation, tangent1), -0.001, 0.001) * idt

        if compute_effective_mass:
            cc_set_eff_n(contacts_state, contact, effective_mass[0])
            cc_set_eff_t1(contacts_state, contact, effective_mass[1])
            cc_set_eff_t2(contacts_state, contact, effective_mass[2])
        cc_set_bias(contacts_state, contact, bias)
        cc_set_bias_t1(contacts_state, contact, bias_t0)
        cc_set_bias_t2(contacts_state, contact, bias_t1)
        cc_set_r0(contacts_state, contact, contact_point - bodies.position[body0])
        cc_set_r1(contacts_state, contact, contact_point - bodies.position[body1])

        if apply_warmstart:
            impulse = (
                cc_get_normal_lambda(contacts_state, contact) * normal
                + cc_get_tangent1_lambda(contacts_state, contact) * tangent0
                + cc_get_tangent2_lambda(contacts_state, contact) * tangent1
            )
            if use_deferred:
                _apply_deferred_impulse(bodies, body0, contact_point, body1, contact_point, impulse)
            else:
                _apply_pair_impulse(bodies, body0, contact_point, body1, contact_point, impulse)


@wp.func
def reduced_contact_iterate(
    columns: ContactColumnContainer,
    column: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    contacts_state: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    use_deferred: wp.bool,
):
    """Run one sequential contact-manifold PGS sweep."""
    body0 = contact_get_body1(columns, column)
    body1 = contact_get_body2(columns, column)
    first = contact_get_contact_first(columns, column)
    count = contact_get_contact_count(columns, column)
    friction_static = contact_get_friction(columns, column)
    friction_dynamic = contact_get_friction_dynamic(columns, column)
    mass_coeff = wp.float32(1.0)
    impulse_coeff = wp.float32(0.0)
    if use_bias:
        _bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
            DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt
        )

    for offset in range(count):
        contact = first + offset
        normal = cc_get_normal(contacts_state, contact)
        tangent0 = cc_get_tangent1(contacts_state, contact)
        tangent1 = wp.cross(normal, tangent0)
        local0 = cc_get_local_p0(contacts_state, contact)
        local1 = cc_get_local_p1(contacts_state, contact)
        point0 = (
            bodies.position[body0]
            + wp.quat_rotate(bodies.orientation[body0], local0 - bodies.body_com[body0])
            + contacts.rigid_contact_margin0[contact] * normal
        )
        point1 = (
            bodies.position[body1]
            + wp.quat_rotate(bodies.orientation[body1], local1 - bodies.body_com[body1])
            - contacts.rigid_contact_margin1[contact] * normal
        )
        contact_point = wp.float32(0.5) * (point0 + point1)
        if use_deferred:
            relative_velocity = _deferred_point_velocity(bodies, body1, contact_point) - _deferred_point_velocity(
                bodies, body0, contact_point
            )
        else:
            relative_velocity = _point_velocity(bodies, body1, contact_point) - _point_velocity(
                bodies, body0, contact_point
            )
        bias = cc_get_bias(contacts_state, contact)
        speculative = bias > wp.float32(0.0)
        if speculative and not use_bias:
            continue
        bias_t0 = cc_get_bias_t1(contacts_state, contact) if use_bias else wp.float32(0.0)
        bias_t1 = cc_get_bias_t2(contacts_state, contact) if use_bias else wp.float32(0.0)
        if not use_bias:
            bias = wp.float32(0.0)

        row_mass_coeff = mass_coeff
        row_impulse_coeff = impulse_coeff
        mu_static = friction_static
        mu_dynamic = friction_dynamic
        if speculative:
            row_mass_coeff = wp.float32(1.0)
            row_impulse_coeff = wp.float32(0.0)
            if bias > idt * wp.float32(0.002):
                mu_static = wp.float32(0.0)
                mu_dynamic = wp.float32(0.0)

        impulse = contact_project_velocity_update_no_soft_pd(
            contacts_state,
            contact,
            normal,
            tangent0,
            tangent1,
            wp.dot(relative_velocity, normal),
            wp.dot(relative_velocity, tangent0),
            wp.dot(relative_velocity, tangent1),
            cc_get_eff_n(contacts_state, contact),
            cc_get_eff_t1(contacts_state, contact),
            cc_get_eff_t2(contacts_state, contact),
            bias,
            bias_t0,
            bias_t1,
            mu_static,
            mu_dynamic,
            row_mass_coeff,
            row_impulse_coeff,
            sor_boost,
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
        )
        if use_deferred:
            _apply_deferred_impulse(bodies, body0, contact_point, body1, contact_point, impulse)
        else:
            _apply_pair_impulse(bodies, body0, contact_point, body1, contact_point, impulse)


__all__ = ["reduced_contact_deferred_owner", "reduced_contact_iterate", "reduced_contact_prepare"]
