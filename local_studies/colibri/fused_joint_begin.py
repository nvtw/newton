# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Local fused joint preparation; helpers preserve source expressions exactly."""

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations import direct_equality as _direct
from newton._src.solvers.phoenx.articulations.block_joint_system import BlockJointSystem

for _name, _value in vars(_direct).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

@wp.func
def _prepare_at(_index: wp.int32, structural_joints: wp.array[wp.int32], effective_joint_mode: wp.array[wp.int32], effective_joint_axis: wp.array[wp.vec3], generic_linear_axes: wp.array[wp.vec3], generic_angular_axes: wp.array[wp.vec3], generic_linear_count: wp.array[wp.int32], generic_angular_count: wp.array[wp.int32], joint_parent: wp.array[wp.int32], joint_child: wp.array[wp.int32], joint_qd_start: wp.array[wp.int32], joint_dof_dim: wp.array2d[wp.int32], joint_x_p: wp.array[wp.transform], joint_x_c: wp.array[wp.transform], cable_rest_relative_orientation: wp.array[wp.quat], joint_target_ke: wp.array[wp.float32], joint_target_kd: wp.array[wp.float32], bodies: BodyContainer, idt: wp.float32, row_count: wp.array[wp.int32], row_wrench0: wp.array2d[wp.spatial_vector], row_wrench1: wp.array2d[wp.spatial_vector], row_bias: wp.array2d[wp.float32], row_error: wp.array2d[wp.float32], row_stiffness: wp.array2d[wp.float32], row_damping: wp.array2d[wp.float32]):
    structural_index = _index
    bias_rate, _mass_coeff, _impulse_coeff = soft_constraint_coefficients(DEFAULT_HERTZ_LINEAR, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt)
    joint = structural_joints[structural_index]
    count = _prepare_direct_rows(structural_index, joint, effective_joint_mode, effective_joint_axis, generic_linear_axes, generic_angular_axes, generic_linear_count, generic_angular_count, joint_parent, joint_child, joint_qd_start, joint_dof_dim, joint_x_p, joint_x_c, cable_rest_relative_orientation, joint_target_ke, joint_target_kd, bodies, bias_rate, row_wrench0, row_wrench1, row_bias, row_error, row_stiffness, row_damping)
    row_count[structural_index] = count

@wp.func
def _snapshot_at(_index: wp.int32, row_joint: wp.array[wp.int32], row_local: wp.array[wp.int32], row_dynamic: wp.array[wp.bool], row_dof: wp.array[wp.int32], row_direct_drive: wp.array[wp.bool], joint_to_structural: wp.array[wp.int32], effective_joint_mode: wp.array[wp.int32], effective_joint_axis: wp.array[wp.vec3], joint_type: wp.array[wp.int32], joint_qd_start: wp.array[wp.int32], joint_dof_dim: wp.array2d[wp.int32], row_target_q: wp.array[wp.int32], joint_parent: wp.array[wp.int32], joint_child: wp.array[wp.int32], joint_x_p: wp.array[wp.transform], joint_x_c: wp.array[wp.transform], row_wrench0: wp.array2d[wp.spatial_vector], row_wrench1: wp.array2d[wp.spatial_vector], joint_armature: wp.array[wp.float32], joint_damping: wp.array[wp.float32], joint_gear: wp.array[wp.float32], joint_target_mode: wp.array[wp.int32], joint_target_ke: wp.array[wp.float32], joint_target_kd: wp.array[wp.float32], control_target_q: wp.array[wp.float32], control_target_qd: wp.array[wp.float32], dt: wp.float32, bodies: BodyContainer, previous_coordinate: wp.array[wp.float32], coordinate_revolutions: wp.array[wp.int32], dynamic_mass: wp.array[wp.float32], dynamic_old_velocity: wp.array[wp.float32], dynamic_coordinate: wp.array[wp.float32], velocity_reference: wp.array[wp.float32], accumulated_impulse: wp.array[wp.float32], drive_saturated: wp.array[wp.bool]):
    row = _index
    accumulated_impulse[row] = wp.float32(0.0)
    drive_saturated[row] = False
    if not row_dynamic[row]:
        dynamic_mass[row] = wp.float32(0.0)
        dynamic_old_velocity[row] = wp.float32(0.0)
        dynamic_coordinate[row] = wp.float32(0.0)
        velocity_reference[row] = wp.float32(0.0)
        return
    joint = row_joint[row]
    structural_index = joint_to_structural[joint]
    local_row = row_local[row]
    body0 = joint_parent[joint] + wp.int32(1)
    body1 = joint_child[joint] + wp.int32(1)
    relative_velocity = wp.float32(0.0)
    if body0 > wp.int32(0):
        relative_velocity += wp.dot(row_wrench0[structural_index, local_row], _body_com_twist(bodies, body0))
    if body1 > wp.int32(0):
        relative_velocity += wp.dot(row_wrench1[structural_index, local_row], _body_com_twist(bodies, body1))
    dynamic_old_velocity[row] = relative_velocity
    dof = row_dof[row]
    gear = joint_gear[dof]
    armature = joint_armature[dof] * gear * gear
    passive_damping = joint_damping[dof]
    mass = armature + dt * passive_damping
    momentum = armature * relative_velocity
    if row_direct_drive[row]:
        target_mode = joint_target_mode[dof]
        stiffness = joint_target_ke[dof]
        drive_damping = joint_target_kd[dof]
        target_position = control_target_q[row_target_q[row]]
        target_velocity = control_target_qd[dof]
        if target_mode == JointTargetMode.POSITION:
            target_velocity = wp.float32(0.0)
        elif target_mode == JointTargetMode.VELOCITY:
            stiffness = wp.float32(0.0)
        elif target_mode == JointTargetMode.NONE or target_mode == JointTargetMode.EFFORT:
            stiffness = wp.float32(0.0)
            drive_damping = wp.float32(0.0)
            target_velocity = wp.float32(0.0)
        mode = effective_joint_mode[joint]
        x_wpj = _body_origin_transform(bodies, body0) * joint_x_p[joint]
        x_wcj = _body_origin_transform(bodies, body1) * joint_x_c[joint]
        point0 = wp.transform_get_translation(x_wpj)
        point1 = wp.transform_get_translation(x_wcj)
        q0 = wp.transform_get_rotation(x_wpj)
        q1 = wp.transform_get_rotation(x_wcj)
        axis = wp.normalize(wp.quat_rotate(q0, effective_joint_axis[joint]))
        coordinate = dynamic_coordinate[row]
        qd_start = joint_qd_start[joint]
        linear_count = joint_dof_dim[joint, 0]
        angular_count = joint_dof_dim[joint, 1]
        is_single_d6_angular = joint_type[joint] == JointType.D6 and angular_count == wp.int32(1) and (dof >= qd_start + linear_count)
        if mode == JOINT_MODE_REVOLUTE or is_single_d6_angular:
            wrapped = extract_rotation_angle(q1 * wp.quat_inverse(q0), axis)
            counter, previous = revolution_tracker_update(wrapped, coordinate_revolutions[row], previous_coordinate[row])
            coordinate_revolutions[row] = counter
            previous_coordinate[row] = previous
            coordinate = revolution_tracker_angle(counter, previous)
        elif mode == JOINT_MODE_PRISMATIC:
            coordinate = wp.dot(axis, point1 - point0)
        dynamic_coordinate[row] = coordinate
        mass += dt * drive_damping + dt * dt * stiffness
        momentum += dt * (stiffness * (target_position - coordinate) + drive_damping * target_velocity)
    dynamic_mass[row] = wp.max(mass, wp.float32(1e-10))
    velocity_reference[row] = momentum / dynamic_mass[row]

@wp.kernel(enable_backward=False)
def fused_begin(row_starts: wp.array[wp.int32], row_ids: wp.array[wp.int32], p_structural_joints: wp.array[wp.int32], p_effective_joint_mode: wp.array[wp.int32], p_effective_joint_axis: wp.array[wp.vec3], p_generic_linear_axes: wp.array[wp.vec3], p_generic_angular_axes: wp.array[wp.vec3], p_generic_linear_count: wp.array[wp.int32], p_generic_angular_count: wp.array[wp.int32], p_joint_parent: wp.array[wp.int32], p_joint_child: wp.array[wp.int32], p_joint_qd_start: wp.array[wp.int32], p_joint_dof_dim: wp.array2d[wp.int32], p_joint_x_p: wp.array[wp.transform], p_joint_x_c: wp.array[wp.transform], p_cable_rest_relative_orientation: wp.array[wp.quat], p_joint_target_ke: wp.array[wp.float32], p_joint_target_kd: wp.array[wp.float32], p_bodies: BodyContainer, p_idt: wp.float32, p_row_count: wp.array[wp.int32], p_row_wrench0: wp.array2d[wp.spatial_vector], p_row_wrench1: wp.array2d[wp.spatial_vector], p_row_bias: wp.array2d[wp.float32], p_row_error: wp.array2d[wp.float32], p_row_stiffness: wp.array2d[wp.float32], p_row_damping: wp.array2d[wp.float32], s_row_joint: wp.array[wp.int32], s_row_local: wp.array[wp.int32], s_row_dynamic: wp.array[wp.bool], s_row_dof: wp.array[wp.int32], s_row_direct_drive: wp.array[wp.bool], s_joint_to_structural: wp.array[wp.int32], s_effective_joint_mode: wp.array[wp.int32], s_effective_joint_axis: wp.array[wp.vec3], s_joint_type: wp.array[wp.int32], s_joint_qd_start: wp.array[wp.int32], s_joint_dof_dim: wp.array2d[wp.int32], s_row_target_q: wp.array[wp.int32], s_joint_parent: wp.array[wp.int32], s_joint_child: wp.array[wp.int32], s_joint_x_p: wp.array[wp.transform], s_joint_x_c: wp.array[wp.transform], s_row_wrench0: wp.array2d[wp.spatial_vector], s_row_wrench1: wp.array2d[wp.spatial_vector], s_joint_armature: wp.array[wp.float32], s_joint_damping: wp.array[wp.float32], s_joint_gear: wp.array[wp.float32], s_joint_target_mode: wp.array[wp.int32], s_joint_target_ke: wp.array[wp.float32], s_joint_target_kd: wp.array[wp.float32], s_control_target_q: wp.array[wp.float32], s_control_target_qd: wp.array[wp.float32], s_dt: wp.float32, s_bodies: BodyContainer, s_previous_coordinate: wp.array[wp.float32], s_coordinate_revolutions: wp.array[wp.int32], s_dynamic_mass: wp.array[wp.float32], s_dynamic_old_velocity: wp.array[wp.float32], s_dynamic_coordinate: wp.array[wp.float32], s_velocity_reference: wp.array[wp.float32], s_accumulated_impulse: wp.array[wp.float32], s_drive_saturated: wp.array[wp.bool]):
    index = wp.tid()
    _prepare_at(index, p_structural_joints, p_effective_joint_mode, p_effective_joint_axis, p_generic_linear_axes, p_generic_angular_axes, p_generic_linear_count, p_generic_angular_count, p_joint_parent, p_joint_child, p_joint_qd_start, p_joint_dof_dim, p_joint_x_p, p_joint_x_c, p_cable_rest_relative_orientation, p_joint_target_ke, p_joint_target_kd, p_bodies, p_idt, p_row_count, p_row_wrench0, p_row_wrench1, p_row_bias, p_row_error, p_row_stiffness, p_row_damping)
    for pointer in range(row_starts[index], row_starts[index + 1]):
        _snapshot_at(row_ids[pointer], s_row_joint, s_row_local, s_row_dynamic, s_row_dof, s_row_direct_drive, s_joint_to_structural, s_effective_joint_mode, s_effective_joint_axis, s_joint_type, s_joint_qd_start, s_joint_dof_dim, s_row_target_q, s_joint_parent, s_joint_child, s_joint_x_p, s_joint_x_c, s_row_wrench0, s_row_wrench1, s_joint_armature, s_joint_damping, s_joint_gear, s_joint_target_mode, s_joint_target_ke, s_joint_target_kd, s_control_target_q, s_control_target_qd, s_dt, s_bodies, s_previous_coordinate, s_coordinate_revolutions, s_dynamic_mass, s_dynamic_old_velocity, s_dynamic_coordinate, s_velocity_reference, s_accumulated_impulse, s_drive_saturated)

_original_init = BlockJointSystem.__init__
_original_begin = BlockJointSystem.begin_substep


def _init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    if not self.enabled or self.has_multi_axis_dynamic_rows:
        return
    rows = {int(joint): [] for joint in self.topology.joints}
    for row, joint in enumerate(self.topology.row_joint):
        rows[int(joint)].append(row)
    starts, ids = [0], []
    for joint in self.topology.joints:
        ids.extend(rows[int(joint)])
        starts.append(len(ids))
    assert sorted(ids) == list(range(len(self.topology.row_joint)))
    self._fused_starts = wp.array(starts, dtype=wp.int32, device=self.model.device)
    self._fused_ids = wp.array(ids, dtype=wp.int32, device=self.model.device)
    self._fused_source_rows = self.row_joint


def _begin(self, idt):
    if (
        not self.enabled
        or self.has_multi_axis_dynamic_rows
        or not hasattr(self, "_fused_starts")
        or self.row_joint is not self._fused_source_rows
    ):
        return _original_begin(self, idt)
    recorded = []
    launch = wp.launch

    def record(kernel, *args, **kwargs):
        assert not args
        recorded.append((kernel, kwargs))

    wp.launch = record
    try:
        _original_begin(self, idt)
    finally:
        wp.launch = launch
    assert [item[0] for item in recorded] == [
        _direct._prepare_direct_equality_rows_kernel,
        _direct._snapshot_direct_dynamic_velocity_kernel,
    ]
    launch(
        fused_begin,
        dim=len(self.topology.joints),
        inputs=[self._fused_starts, self._fused_ids, *recorded[0][1]["inputs"], *recorded[1][1]["inputs"]],
        device=self.model.device,
    )


if __name__ == "__main__":
    import runpy

    BlockJointSystem.__init__ = _init
    BlockJointSystem.begin_substep = _begin
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
