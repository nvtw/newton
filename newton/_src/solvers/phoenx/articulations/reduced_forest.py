# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Batched physical inverse mass for a small forest of reduced trees."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.sim.enums import BodyFlags
from newton._src.solvers.phoenx.articulations.reduced_contact_block import _broadcast_contact_scalar
from newton._src.solvers.phoenx.body import BodyContainer
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
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_bias,
    cc_set_eff_n,
    cc_set_eff_t1,
    cc_set_eff_t2,
    cc_set_normal_lambda,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_coupled_velocity_update_no_soft_pd

from .reduced_contact import _point_velocity, _prepare_contact_bias_geometry, _rigid_endpoint_inverse_mass

_vec6 = wp.types.vector(length=6, dtype=wp.float32)
_MAX_JOINT_DOF = 6


@wp.kernel(enable_backward=False)
def _build_forest_inverse_mass(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_s: wp.array[wp.spatial_vector],
    joint_u_matrix: wp.array[wp.spatial_vector],
    joint_d_inv: wp.array2d[wp.float32],
    compact_dofs: wp.array[wp.int32],
    global_to_compact: wp.array[wp.int32],
    body_work: wp.array2d[wp.spatial_vector],
    joint_work: wp.array2d[wp.float32],
    body_acceleration: wp.array2d[wp.spatial_vector],
    inverse_mass: wp.array2d[wp.float32],
):
    articulation, basis = wp.tid()
    basis_dof = compact_dofs[basis]
    start = articulation_start[articulation]
    end = articulation_end[articulation]
    if basis_dof < joint_qd_start[start] or basis_dof >= joint_qd_start[end]:
        return

    for joint in range(start, end):
        child = joint_child[joint]
        body_work[child, basis] = wp.spatial_vector()
        body_acceleration[child, basis] = wp.spatial_vector()

    for reverse in range(end - start):
        joint = end - wp.int32(1) - reverse
        parent = joint_parent[joint]
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        p = body_work[child, basis]
        reduced_force = _vec6(0.0)
        d_inv_u = _vec6(0.0)

        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                reduced_force[row] = (wp.float32(1.0) if dof == basis_dof else wp.float32(0.0)) - wp.dot(
                    joint_s[dof], p
                )
                joint_work[dof, basis] = reduced_force[row]
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                for column in range(_MAX_JOINT_DOF):
                    if wp.int32(column) < dof_count:
                        d_inv_u[row] += joint_d_inv[dof_start + wp.int32(row), column] * reduced_force[column]

        propagated = p
        for column in range(_MAX_JOINT_DOF):
            if wp.int32(column) < dof_count:
                propagated += joint_u_matrix[dof_start + wp.int32(column)] * d_inv_u[column]
        if parent >= wp.int32(0):
            body_work[parent, basis] = body_work[parent, basis] + propagated

    for joint in range(start, end):
        parent = joint_parent[joint]
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        parent_acceleration = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_acceleration = body_acceleration[parent, basis]

        rhs = _vec6(0.0)
        qdd = _vec6(0.0)
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                rhs[row] = joint_work[dof, basis] - wp.dot(joint_u_matrix[dof], parent_acceleration)
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                for column in range(_MAX_JOINT_DOF):
                    if wp.int32(column) < dof_count:
                        qdd[row] += joint_d_inv[dof_start + wp.int32(row), column] * rhs[column]

        child_acceleration = parent_acceleration
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                inverse_mass[global_to_compact[dof], basis] = qdd[row]
                child_acceleration += joint_s[dof] * qdd[row]
        body_acceleration[child, basis] = child_acceleration


class ForestInverseMass:
    """Apply independent tree factors to all compact unit forces in parallel."""

    def __init__(self, system):
        self.system = system
        self.model = system.model
        starts = self.model.articulation_start.numpy()
        ends = self.model.articulation_end.numpy()
        qd_starts = self.model.joint_qd_start.numpy()
        compact = np.concatenate(
            [
                np.arange(qd_starts[start], qd_starts[end], dtype=np.int32)
                for start, end in zip(starts, ends, strict=False)
            ]
        )
        self.dof_count = int(compact.size)
        self.width = 1 << max(0, self.dof_count - 1).bit_length()
        if self.dof_count > 64:
            raise ValueError("The forest contact prototype supports at most 64 generalized coordinates")
        inverse = np.full(self.model.joint_dof_count, -1, dtype=np.int32)
        inverse[compact] = np.arange(self.dof_count, dtype=np.int32)
        device = self.model.device
        self.compact_dofs = wp.array(compact, dtype=wp.int32, device=device)
        self.global_to_compact = wp.array(inverse, dtype=wp.int32, device=device)
        self.matrix = wp.zeros((self.width, self.width), dtype=wp.float32, device=device)
        self.body_work = wp.zeros((self.model.body_count, self.dof_count), dtype=wp.spatial_vector, device=device)
        self.body_acceleration = wp.zeros_like(self.body_work)
        self.joint_work = wp.zeros((self.model.joint_dof_count, self.dof_count), dtype=wp.float32, device=device)

    def refresh(self):
        system = self.system
        model = self.model
        wp.launch(
            _build_forest_inverse_mass,
            dim=(model.articulation_count, self.dof_count),
            inputs=[
                model.articulation_start,
                model.articulation_end,
                model.joint_parent,
                model.joint_child,
                model.joint_qd_start,
                system.joint_s,
                system.joint_u_matrix,
                system.joint_d_inv,
                self.compact_dofs,
                self.global_to_compact,
                self.body_work,
                self.joint_work,
                self.body_acceleration,
            ],
            outputs=[self.matrix],
            device=model.device,
        )


@wp.kernel(enable_backward=False)
def _map_forest_contacts(
    columns: ContactColumnContainer,
    num_columns: wp.array[wp.int32],
    contact_column: wp.array[wp.int32],
):
    column = wp.tid()
    if column < num_columns[0]:
        first = contact_get_contact_first(columns, column)
        count = contact_get_contact_count(columns, column)
        for contact in range(first, first + count):
            contact_column[contact] = column


@wp.kernel(enable_backward=False)
def _build_forest_contact_rows(
    bodies: BodyContainer,
    columns: ContactColumnContainer,
    contacts: ContactViews,
    cc: ContactContainer,
    contact_column: wp.array[wp.int32],
    compact_dofs: wp.array[wp.int32],
    dof_count: wp.int32,
    idt: wp.float32,
    jacobian: wp.array2d[wp.float32],
    prescribed_velocity: wp.array[wp.vec3],
    unconstrained_normal: wp.array[wp.vec3],
):
    contact, compact = wp.tid()
    if contact >= contacts.rigid_contact_count[0] or compact >= dof_count:
        return
    data = bodies.reduced
    column = contact_column[contact]
    body0 = contact_get_body1(columns, column)
    body1 = contact_get_body2(columns, column)
    normal = cc_get_normal(cc, contact)
    tangent0 = cc_get_tangent1(cc, contact)
    tangent1 = wp.cross(normal, tangent0)
    point0 = (
        bodies.position[body0]
        + wp.quat_rotate(bodies.orientation[body0], contacts.rigid_contact_point0[contact] - bodies.body_com[body0])
        + contacts.rigid_contact_margin0[contact] * normal
    )
    point1 = (
        bodies.position[body1]
        + wp.quat_rotate(bodies.orientation[body1], contacts.rigid_contact_point1[contact] - bodies.body_com[body1])
        - contacts.rigid_contact_margin1[contact] * normal
    )
    point = wp.float32(0.5) * (point0 + point1)
    dof = compact_dofs[compact]
    for axis in range(3):
        direction = normal
        if axis == 1:
            direction = tangent0
        elif axis == 2:
            direction = tangent1
        wrench = wp.spatial_vector()
        for side in range(2):
            slot = body0
            sign = wp.float32(-1.0)
            if side == 1:
                slot = body1
                sign = wp.float32(1.0)
            articulation = data.body_articulation[slot]
            if articulation >= wp.int32(0):
                body = slot - wp.int32(1)
                for path in range(data.body_path_start[body], data.body_path_start[body + wp.int32(1)]):
                    joint = data.body_path_joint[path]
                    if dof >= data.joint_qd_start[joint] and dof < data.joint_qd_start[joint + wp.int32(1)]:
                        force = sign * direction
                        wrench += wp.spatial_vector(
                            force, wp.cross(point - data.articulation_origin[articulation], force)
                        )
        jacobian[3 * contact + axis, compact] = wp.dot(data.joint_s[dof], wrench)
    if compact == wp.int32(0):
        bias_rate, _mass, _impulse = soft_constraint_coefficients(
            DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt
        )
        _prepare_contact_bias_geometry(
            bodies,
            body0,
            body1,
            wp.bool(False),
            cc,
            contacts,
            contact,
            idt,
            bias_rate,
            normal,
            tangent0,
            tangent1,
            point1 - point0,
            point - bodies.position[body0],
            point - bodies.position[body1],
        )
        prescribed = wp.vec3(0.0)
        if data.body_articulation[body0] < wp.int32(0):
            prescribed -= _point_velocity(bodies, body0, point)
        if data.body_articulation[body1] < wp.int32(0):
            prescribed += _point_velocity(bodies, body1, point)
        prescribed_velocity[contact] = wp.vec3(
            wp.dot(normal, prescribed), wp.dot(tangent0, prescribed), wp.dot(tangent1, prescribed)
        )
        mobility = wp.vec3(0.0)
        for axis in range(3):
            direction = normal
            if axis == 1:
                direction = tangent0
            elif axis == 2:
                direction = tangent1
            mobility[axis] = _rigid_endpoint_inverse_mass(
                bodies, body0, point, direction
            ) + _rigid_endpoint_inverse_mass(bodies, body1, point, direction)
        unconstrained_normal[contact] = mobility


@wp.kernel(enable_backward=False)
def _apply_forest_mass(
    contact_count: wp.array[wp.int32],
    dof_count: wp.int32,
    inverse_mass: wp.array2d[wp.float32],
    jacobian: wp.array2d[wp.float32],
    response: wp.array2d[wp.float32],
):
    row, target = wp.tid()
    if row >= wp.int32(3) * contact_count[0] or target >= dof_count:
        return
    value = wp.float32(0.0)
    for source in range(dof_count):
        value += inverse_mass[target, source] * jacobian[row, source]
    response[row, target] = value


@wp.kernel(enable_backward=False)
def _prepare_forest_mobility(
    contact_count: wp.array[wp.int32],
    dof_count: wp.int32,
    jacobian: wp.array2d[wp.float32],
    response: wp.array2d[wp.float32],
    unconstrained: wp.array[wp.vec3],
    cc: ContactContainer,
    cross_mobility: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    diagonal = wp.vec3(0.0)
    cross = wp.vec3(0.0)
    for axis in range(3):
        value = wp.float64(0.0)
        for dof in range(dof_count):
            value += wp.float64(jacobian[3 * contact + axis, dof]) * wp.float64(response[3 * contact + axis, dof])
        diagonal[axis] = wp.float32(value)
    for axis in range(3):
        first = wp.int32(0)
        second = wp.int32(1)
        if axis == 1:
            second = wp.int32(2)
        elif axis == 2:
            first = wp.int32(1)
            second = wp.int32(2)
        value = wp.float64(0.0)
        for dof in range(dof_count):
            value += wp.float64(jacobian[3 * contact + first, dof]) * wp.float64(response[3 * contact + second, dof])
        cross[axis] = wp.float32(value)
    cross_mobility[contact] = cross
    effective = wp.vec3(0.0)
    for axis in range(3):
        if diagonal[axis] > wp.max(
            wp.float32(1.0e-12), wp.float32(5.820766091346741e-11) * unconstrained[contact][axis]
        ):
            effective[axis] = wp.float32(1.0) / diagonal[axis]
    cc_set_eff_n(cc, contact, effective[0])
    cc_set_eff_t1(cc, contact, effective[1])
    cc_set_eff_t2(cc, contact, effective[2])
    if effective[0] == wp.float32(0.0):
        cc_set_normal_lambda(cc, contact, wp.float32(0.0))
    if effective[1] == wp.float32(0.0):
        cc_set_tangent1_lambda(cc, contact, wp.float32(0.0))
    if effective[2] == wp.float32(0.0):
        cc_set_tangent2_lambda(cc, contact, wp.float32(0.0))
    bias = cc_get_bias(cc, contact)
    if bias < wp.float32(0.0) and unconstrained[contact][0] > wp.float32(1.0e-12):
        cc_set_bias(
            cc, contact, bias * wp.clamp(diagonal[0] / unconstrained[contact][0], wp.float32(0.0), wp.float32(1.0))
        )


@wp.kernel(enable_backward=False)
def _gather_forest_velocity(
    bodies: BodyContainer,
    compact_dofs: wp.array[wp.int32],
    velocity: wp.array[wp.float32],
):
    compact = wp.tid()
    velocity[compact] = bodies.reduced.joint_qd[compact_dofs[compact]]


@wp.kernel(enable_backward=False)
def _publish_forest_velocity(
    bodies: BodyContainer,
    global_to_compact: wp.array[wp.int32],
    velocity: wp.array[wp.float32],
):
    articulation = wp.tid()
    data = bodies.reduced
    start = data.articulation_start[articulation]
    end = data.articulation_end[articulation]
    for joint in range(start, end):
        parent = data.joint_parent[joint]
        child = data.joint_child[joint]
        twist = wp.spatial_vector()
        if parent >= wp.int32(0):
            twist = data.body_acceleration[parent]
        for dof in range(data.joint_qd_start[joint], data.joint_qd_start[joint + wp.int32(1)]):
            speed = velocity[global_to_compact[dof]]
            data.joint_qd[dof] = speed
            twist += data.joint_s[dof] * speed
        data.body_acceleration[child] = twist
        omega = wp.spatial_bottom(twist)
        bodies.angular_velocity[child + wp.int32(1)] = omega
        bodies.velocity[child + wp.int32(1)] = wp.spatial_top(twist) + wp.cross(
            omega, wp.transform_get_translation(data.body_q_com[child])
        )


def _make_forest_contact_solve(width):
    @wp.kernel(enable_backward=False, module=wp.get_module(f"reduced_forest_solve_{width}"))
    def solve(
        contacts: ContactViews,
        columns: ContactColumnContainer,
        cc: ContactContainer,
        contact_column: wp.array[wp.int32],
        jacobian: wp.array2d[wp.float32],
        response: wp.array2d[wp.float32],
        prescribed_velocity: wp.array[wp.vec3],
        cross_mobility: wp.array[wp.vec3],
        velocity: wp.array[wp.float32],
        idt: wp.float32,
        iterations: wp.int32,
        use_bias: wp.bool,
        warmstart: wp.bool,
        sor: wp.float32,
    ):
        lane = wp.tid()
        initial = wp.tile_load(velocity, shape=width, storage="register")
        delta = wp.tile_zeros(shape=width, dtype=wp.float32, storage="register")
        # Collision counters can report overflow beyond stored contacts.
        count = wp.min(contacts.rigid_contact_count[0], contact_column.shape[0])
        if warmstart:
            for contact in range(count):
                for axis in range(3):
                    load = wp.float32(0.0)
                    if lane == wp.int32(0):
                        if axis == 0:
                            load = cc_get_normal_lambda(cc, contact)
                        elif axis == 1:
                            load = cc_get_tangent1_lambda(cc, contact)
                        else:
                            load = cc_get_tangent2_lambda(cc, contact)
                    load = _broadcast_contact_scalar(load)
                    row_response = wp.tile_load(response[3 * contact + axis], shape=width, storage="register")
                    delta += load * row_response
        mass_coeff = wp.float32(1.0)
        impulse_coeff = wp.float32(0.0)
        if use_bias:
            _bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
                DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt
            )
        for iteration in range(iterations):
            for offset in range(count):
                contact = offset
                if (iteration & wp.int32(1)) != wp.int32(0):
                    contact = count - wp.int32(1) - offset
                j0 = wp.tile_load(jacobian[3 * contact], shape=width, storage="register")
                j1 = wp.tile_load(jacobian[3 * contact + 1], shape=width, storage="register")
                j2 = wp.tile_load(jacobian[3 * contact + 2], shape=width, storage="register")
                r0 = wp.tile_load(response[3 * contact], shape=width, storage="register")
                r1 = wp.tile_load(response[3 * contact + 1], shape=width, storage="register")
                r2 = wp.tile_load(response[3 * contact + 2], shape=width, storage="register")
                current = initial + delta
                prescribed = prescribed_velocity[contact]
                v0 = prescribed[0] + wp.tile_extract(wp.tile_sum(j0 * current), 0)
                v1 = prescribed[1] + wp.tile_extract(wp.tile_sum(j1 * current), 0)
                v2 = prescribed[2] + wp.tile_extract(wp.tile_sum(j2 * current), 0)
                dn = wp.float32(0.0)
                dt1 = wp.float32(0.0)
                dt2 = wp.float32(0.0)
                if lane == wp.int32(0):
                    bias = cc_get_bias(cc, contact)
                    speculative = bias > wp.float32(0.0)
                    if not (speculative and not use_bias):
                        tangent_bias1 = cc_get_bias_t1(cc, contact) if use_bias else wp.float32(0.0)
                        tangent_bias2 = cc_get_bias_t2(cc, contact) if use_bias else wp.float32(0.0)
                        if not use_bias:
                            bias = wp.float32(0.0)
                        column = contact_column[contact]
                        mu_s = contact_get_friction(columns, column)
                        mu_k = contact_get_friction_dynamic(columns, column)
                        row_mass = mass_coeff
                        row_impulse = impulse_coeff
                        if speculative:
                            row_mass = wp.float32(1.0)
                            row_impulse = wp.float32(0.0)
                            if bias > wp.float32(0.002) * idt:
                                mu_s = wp.float32(0.0)
                                mu_k = wp.float32(0.0)
                        n = cc_get_normal(cc, contact)
                        t1 = cc_get_tangent1(cc, contact)
                        t2 = wp.cross(n, t1)
                        cross = cross_mobility[contact]
                        impulse = contact_project_coupled_velocity_update_no_soft_pd(
                            cc,
                            contact,
                            n,
                            t1,
                            t2,
                            v0,
                            v1,
                            v2,
                            cc_get_eff_n(cc, contact),
                            cc_get_eff_t1(cc, contact),
                            cc_get_eff_t2(cc, contact),
                            bias,
                            tangent_bias1,
                            tangent_bias2,
                            mu_s,
                            mu_k,
                            row_mass,
                            row_impulse,
                            sor,
                            wp.float32(0.0),
                            wp.float32(0.0),
                            wp.float32(0.0),
                            cross[0],
                            cross[1],
                            cross[2],
                        )
                        dn = wp.dot(impulse, n)
                        dt1 = wp.dot(impulse, t1)
                        dt2 = wp.dot(impulse, t2)
                dn = _broadcast_contact_scalar(dn)
                dt1 = _broadcast_contact_scalar(dt1)
                dt2 = _broadcast_contact_scalar(dt2)
                delta += dn * r0 + dt1 * r1 + dt2 * r2
        wp.tile_store(velocity, initial + delta)

    return solve


_FOREST_SOLVE_KERNELS = {}


class ReducedForestContactSystem:
    """Single-world contact prototype with independently factored physical trees."""

    def __init__(self, backend, contact_capacity):
        self.backend = backend
        self.model = backend.model
        dynamic = (self.model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) == 0
        if np.any(dynamic & ~backend.body_is_reduced_np):
            raise ValueError("The forest prototype requires every dynamic body to belong to a reduced tree")
        if int(self.model.world_count) > 1:
            raise ValueError("The forest prototype requires a single world")
        self.mass = ForestInverseMass(backend.system)
        self.capacity = int(contact_capacity)
        device = self.model.device
        width = self.mass.width
        self.contact_column = wp.zeros(self.capacity, dtype=wp.int32, device=device)
        self.jacobian = wp.zeros((3 * self.capacity, width), dtype=wp.float32, device=device)
        self.response = wp.zeros_like(self.jacobian)
        self.prescribed_velocity = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        self.unconstrained_normal = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        self.cross_mobility = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        self.velocity = wp.zeros(width, dtype=wp.float32, device=device)
        if width not in _FOREST_SOLVE_KERNELS:
            _FOREST_SOLVE_KERNELS[width] = _make_forest_contact_solve(width)
        self.solve_kernel = _FOREST_SOLVE_KERNELS[width]

    def solve(self, world, idt, iterations, *, use_bias, prepare):
        device = self.model.device
        contacts = world._active_contact_views()
        columns = world._contact_cols
        cc = world._contact_container
        self.mass.refresh()
        wp.launch(
            _map_forest_contacts,
            dim=world.max_contact_columns,
            inputs=[columns, world._ingest_scratch.num_contact_columns],
            outputs=[self.contact_column],
            device=device,
        )
        wp.launch(
            _build_forest_contact_rows,
            dim=(self.capacity, self.mass.dof_count),
            inputs=[
                world.bodies,
                columns,
                contacts,
                cc,
                self.contact_column,
                self.mass.compact_dofs,
                wp.int32(self.mass.dof_count),
                idt,
            ],
            outputs=[self.jacobian, self.prescribed_velocity, self.unconstrained_normal],
            device=device,
        )
        wp.launch(
            _apply_forest_mass,
            dim=(3 * self.capacity, self.mass.dof_count),
            inputs=[contacts.rigid_contact_count, wp.int32(self.mass.dof_count), self.mass.matrix, self.jacobian],
            outputs=[self.response],
            device=device,
        )
        wp.launch(
            _prepare_forest_mobility,
            dim=self.capacity,
            inputs=[
                contacts.rigid_contact_count,
                wp.int32(self.mass.dof_count),
                self.jacobian,
                self.response,
                self.unconstrained_normal,
                cc,
            ],
            outputs=[self.cross_mobility],
            device=device,
        )
        wp.launch(
            _gather_forest_velocity,
            dim=self.mass.dof_count,
            inputs=[world.bodies, self.mass.compact_dofs],
            outputs=[self.velocity],
            device=device,
        )
        wp.launch(
            self.solve_kernel,
            dim=32,
            block_dim=32,
            inputs=[
                contacts,
                columns,
                cc,
                self.contact_column,
                self.jacobian,
                self.response,
                self.prescribed_velocity,
                self.cross_mobility,
                self.velocity,
                idt,
                wp.int32(iterations),
                wp.bool(use_bias),
                wp.bool(use_bias and prepare),
                wp.float32(world.sor_boost),
            ],
            device=device,
        )
        wp.launch(
            _publish_forest_velocity,
            dim=self.model.articulation_count,
            inputs=[world.bodies, self.mass.global_to_compact, self.velocity],
            device=device,
        )
