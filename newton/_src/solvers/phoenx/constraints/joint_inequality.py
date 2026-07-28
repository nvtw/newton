# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lean drive, limit, and friction iteration for direct-equality joints."""

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.body import BodyContainer, body_set_access_mode
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    read_int,
    read_vec3,
    write_int,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    _OFF_AXIS_WORLD,
    _OFF_BODY1,
    _OFF_BODY2,
    _OFF_CLAMP,
    _OFF_JOINT_MODE,
    _OFF_R1_B1,
    _OFF_R1_B2,
    _OFF_STRUCTURAL_DIRECT,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
    _axial_drive_limit_iterate,
    _d6_angular_limits_block,
    _ms_load_body_pair,
    _ms_store_body_pair,
)
from newton._src.solvers.phoenx.mass_splitting import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer


@wp.kernel(enable_backward=False)
def mark_direct_equality_joints_kernel(
    constraints: ConstraintContainer,
    structural_direct: wp.array[wp.int32],
    num_joints: wp.int32,
):
    cid = wp.tid()
    if cid < num_joints:
        write_int(constraints, _OFF_STRUCTURAL_DIRECT, cid, structural_direct[cid])


@wp.func
def actuated_double_ball_socket_iterate_inequality(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
):
    """Iterate only the free-axis drive, limit, and friction rows."""
    mode = read_int(constraints, _OFF_JOINT_MODE, cid)
    if (
        mode != JOINT_MODE_REVOLUTE
        and mode != JOINT_MODE_PRISMATIC
        and mode != JOINT_MODE_BALL_SOCKET
        and mode != JOINT_MODE_UNIVERSAL
    ):
        return

    body1 = read_int(constraints, _OFF_BODY1, cid)
    body2 = read_int(constraints, _OFF_BODY2, cid)
    body_set_access_mode(bodies, body1, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_set_access_mode(bodies, body2, ACCESS_MODE_VELOCITY_LEVEL, idt)
    (
        velocity1,
        velocity2,
        angular_velocity1,
        angular_velocity2,
        inverse_mass1,
        inverse_mass2,
        inverse_inertia1,
        inverse_inertia2,
        slot1,
        slot2,
    ) = _ms_load_body_pair(
        bodies,
        particles,
        copy_state,
        body1,
        body2,
        parallel_id,
        num_bodies,
    )

    if mode == JOINT_MODE_BALL_SOCKET or mode == JOINT_MODE_UNIVERSAL:
        angular_velocity1, angular_velocity2 = _d6_angular_limits_block(
            constraints,
            cid,
            wp.int32(0),
            bodies,
            body1,
            mode,
            angular_velocity1,
            angular_velocity2,
            inverse_inertia1,
            inverse_inertia2,
            idt,
            sor_boost,
        )
        _ms_store_body_pair(
            bodies,
            particles,
            copy_state,
            body1,
            body2,
            slot1,
            slot2,
            num_bodies,
            velocity1,
            angular_velocity1,
            velocity2,
            angular_velocity2,
        )
        return

    axis = read_vec3(constraints, _OFF_AXIS_WORLD, cid)
    clamp = read_int(constraints, _OFF_CLAMP, cid)
    if mode == JOINT_MODE_REVOLUTE:
        axial_velocity = wp.dot(axis, angular_velocity1 - angular_velocity2)
        impulse = _axial_drive_limit_iterate(
            constraints,
            cid,
            wp.int32(0),
            axial_velocity,
            clamp,
            idt,
            sor_boost,
            use_bias,
        )
        angular_velocity1 += inverse_inertia1 @ (axis * impulse)
        angular_velocity2 -= inverse_inertia2 @ (axis * impulse)
    else:
        lever1 = read_vec3(constraints, _OFF_R1_B1, cid)
        lever2 = read_vec3(constraints, _OFF_R1_B2, cid)
        anchor_velocity1 = velocity1 + wp.cross(angular_velocity1, lever1)
        anchor_velocity2 = velocity2 + wp.cross(angular_velocity2, lever2)
        axial_velocity = wp.dot(axis, anchor_velocity1 - anchor_velocity2)
        impulse = _axial_drive_limit_iterate(
            constraints,
            cid,
            wp.int32(0),
            axial_velocity,
            clamp,
            idt,
            sor_boost,
            use_bias,
        )
        linear_impulse = axis * impulse
        velocity1 += inverse_mass1 * linear_impulse
        angular_velocity1 += inverse_inertia1 @ wp.cross(lever1, linear_impulse)
        velocity2 -= inverse_mass2 * linear_impulse
        angular_velocity2 -= inverse_inertia2 @ wp.cross(lever2, linear_impulse)

    _ms_store_body_pair(
        bodies,
        particles,
        copy_state,
        body1,
        body2,
        slot1,
        slot2,
        num_bodies,
        velocity1,
        angular_velocity1,
        velocity2,
        angular_velocity2,
    )


__all__ = ["actuated_double_ball_socket_iterate_inequality"]
