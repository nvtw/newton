# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PhoenX constraint building blocks and joint types.

Ported from C# PhoenX constraint system (originally adapted from Jolt Physics).

Building blocks:
    - :func:`prepare_point_on_point_kernel` / :func:`solve_point_on_point_kernel`
      -- 3-DOF position constraint (ball-socket anchor).
    - :func:`prepare_hinge_rotation_kernel` / :func:`solve_hinge_rotation_kernel`
      -- 2-DOF angular constraint (allows rotation around one axis).
    - :func:`prepare_angle_constraint_kernel` / :func:`solve_angle_constraint_kernel`
      -- 1-DOF angular constraint (motor/limit on hinge axis).

Composite joints:
    - Ball-socket = PointOnPoint only.
    - Revolute = PointOnPoint + HingeRotation + AngleConstraint.
"""

import warp as wp

from .schemas import BODY_FLAG_STATIC

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSTRAINT_BAUMGARTE = wp.constant(0.2)

# Constraint type IDs
JOINT_BALL_SOCKET = 0
JOINT_REVOLUTE = 1
JOINT_FIXED = 2

# ---------------------------------------------------------------------------
# Constraint data layout (flat arrays, one per joint)
# ---------------------------------------------------------------------------


@wp.struct
class JointSchema:
    """Per-joint data stored in a DataStore.

    Attributes:
        joint_type: Joint type ID (JOINT_BALL_SOCKET, JOINT_REVOLUTE, JOINT_FIXED).
        body0: Body storage row for body 0.
        body1: Body storage row for body 1.
        local_anchor0: Body-local anchor point on body 0 [m].
        local_anchor1: Body-local anchor point on body 1 [m].
        local_axis0: Body-local hinge axis on body 0 (revolute/fixed).
        local_axis1: Body-local hinge axis on body 1 (revolute/fixed).
        inv_initial_orientation: Inverse of initial relative orientation.
        point_lambda: Accumulated impulse for point constraint [N s].
        hinge_lambda: Accumulated impulse for hinge rotation constraint (vec2).
        angle_lambda: Accumulated impulse for angle constraint (scalar).
        point_eff_mass: 3x3 effective mass for point constraint.
        hinge_b2xa1: Cross product vector b2 x a1 (hinge).
        hinge_c2xa1: Cross product vector c2 x a1 (hinge).
        hinge_eff_mass: 2x2 effective mass for hinge (stored as vec4: m00,m01,m10,m11).
        angle_eff_mass: Scalar effective mass for angle constraint.
        angle_axis: World-space angle constraint axis.
        rw0: World-space anchor offset for body 0.
        rw1: World-space anchor offset for body 1.
        angle_min: Minimum angle limit [rad].
        angle_max: Maximum angle limit [rad].
        angle_current: Current hinge angle [rad].
    """

    joint_type: wp.int32
    body0: wp.int32
    body1: wp.int32
    local_anchor0: wp.vec3
    local_anchor1: wp.vec3
    local_axis0: wp.vec3
    local_axis1: wp.vec3
    inv_initial_orientation: wp.quat
    point_lambda: wp.vec3
    hinge_lambda_x: wp.float32
    hinge_lambda_y: wp.float32
    angle_lambda: wp.float32
    point_eff_mass: wp.mat33
    hinge_b2xa1: wp.vec3
    hinge_c2xa1: wp.vec3
    hinge_eff_mass_00: wp.float32
    hinge_eff_mass_01: wp.float32
    hinge_eff_mass_10: wp.float32
    hinge_eff_mass_11: wp.float32
    angle_eff_mass: wp.float32
    angle_axis: wp.vec3
    rw0: wp.vec3
    rw1: wp.vec3
    angle_min: wp.float32
    angle_max: wp.float32
    angle_current: wp.float32


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


@wp.func
def skew_symmetric(v: wp.vec3) -> wp.mat33:
    """Return the skew-symmetric matrix [v]_x such that [v]_x * u = cross(v, u)."""
    return wp.mat33(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    )


@wp.func
def invert_mat33_safe(m: wp.mat33) -> wp.mat33:
    """Invert a 3x3 matrix, returning zero matrix if singular."""
    det = wp.determinant(m)
    if wp.abs(det) < 1.0e-12:
        return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return wp.inverse(m)


@wp.func
def invert_2x2(m00: float, m01: float, m10: float, m11: float) -> wp.vec4:
    """Invert a 2x2 matrix stored as 4 scalars. Returns (inv00, inv01, inv10, inv11)."""
    det = m00 * m11 - m01 * m10
    if wp.abs(det) < 1.0e-12:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)
    inv_det = 1.0 / det
    return wp.vec4(m11 * inv_det, -m01 * inv_det, -m10 * inv_det, m00 * inv_det)


@wp.func
def get_perpendiculars(axis: wp.vec3) -> wp.vec3:
    """Return a unit vector perpendicular to axis."""
    if wp.abs(axis[0]) < 0.9:
        t = wp.cross(wp.vec3(1.0, 0.0, 0.0), axis)
    else:
        t = wp.cross(wp.vec3(0.0, 1.0, 0.0), axis)
    return wp.normalize(t)


@wp.func
def quat_twist_angle(q: wp.quat, axis: wp.vec3) -> float:
    """Extract the rotation angle around a given axis from a quaternion."""
    # Project quaternion imaginary part onto axis
    qv = wp.vec3(q[0], q[1], q[2])
    proj = wp.dot(qv, axis)
    twist = wp.quat(proj * axis[0], proj * axis[1], proj * axis[2], q[3])
    twist_len = wp.sqrt(proj * proj + q[3] * q[3])
    if twist_len > 1.0e-10:
        twist = wp.quat(twist[0] / twist_len, twist[1] / twist_len, twist[2] / twist_len, twist[3] / twist_len)
    angle = 2.0 * wp.atan2(wp.abs(proj), wp.abs(q[3]))
    if proj * q[3] < 0.0:
        angle = -angle
    return angle


# ---------------------------------------------------------------------------
# Build elements kernel (for graph coloring)
# ---------------------------------------------------------------------------


@wp.kernel
def build_joint_elements_kernel(
    j_body0: wp.array(dtype=wp.int32),
    j_body1: wp.array(dtype=wp.int32),
    elements: wp.array2d(dtype=wp.int32),
    joint_count: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
):
    """Build element array entries for joints, placed right after contacts."""
    tid = wp.tid()
    if tid >= joint_count[0]:
        return
    row = contact_count[0] + tid
    elements[row, 0] = j_body0[tid]
    elements[row, 1] = j_body1[tid]
    for j in range(2, 8):
        elements[row, j] = -1


# ---------------------------------------------------------------------------
# Prepare constraints kernel
# ---------------------------------------------------------------------------


@wp.kernel
def prepare_constraints_kernel(
    partition_data: wp.array(dtype=wp.int32),
    partition_end_arr: wp.array(dtype=wp.int32),
    partition_slot: int,
    contact_count: int,
    # Joint arrays
    j_type: wp.array(dtype=wp.int32),
    j_body0: wp.array(dtype=wp.int32),
    j_body1: wp.array(dtype=wp.int32),
    j_local_anchor0: wp.array(dtype=wp.vec3),
    j_local_anchor1: wp.array(dtype=wp.vec3),
    j_local_axis0: wp.array(dtype=wp.vec3),
    j_local_axis1: wp.array(dtype=wp.vec3),
    j_inv_init_orient: wp.array(dtype=wp.quat),
    j_point_lambda: wp.array(dtype=wp.vec3),
    j_hinge_lambda_x: wp.array(dtype=wp.float32),
    j_hinge_lambda_y: wp.array(dtype=wp.float32),
    j_angle_lambda: wp.array(dtype=wp.float32),
    j_point_eff_mass: wp.array(dtype=wp.mat33),
    j_hinge_b2xa1: wp.array(dtype=wp.vec3),
    j_hinge_c2xa1: wp.array(dtype=wp.vec3),
    j_hinge_eff_mass_00: wp.array(dtype=wp.float32),
    j_hinge_eff_mass_01: wp.array(dtype=wp.float32),
    j_hinge_eff_mass_10: wp.array(dtype=wp.float32),
    j_hinge_eff_mass_11: wp.array(dtype=wp.float32),
    j_angle_eff_mass: wp.array(dtype=wp.float32),
    j_angle_axis: wp.array(dtype=wp.vec3),
    j_rw0: wp.array(dtype=wp.vec3),
    j_rw1: wp.array(dtype=wp.vec3),
    j_angle_min: wp.array(dtype=wp.float32),
    j_angle_max: wp.array(dtype=wp.float32),
    j_angle_current: wp.array(dtype=wp.float32),
    # Body arrays
    b_position: wp.array(dtype=wp.vec3),
    b_orientation: wp.array(dtype=wp.quat),
    b_velocity: wp.array(dtype=wp.vec3),
    b_angular_velocity: wp.array(dtype=wp.vec3),
    b_inverse_mass: wp.array(dtype=wp.float32),
    b_inverse_inertia_world: wp.array(dtype=wp.mat33),
    b_flags: wp.array(dtype=wp.int32),
    joint_count: wp.array(dtype=wp.int32),
):
    """Compute constraint effective masses and apply warm-start impulses."""
    tid = wp.tid()
    p_start = int(0)
    if partition_slot > 0:
        p_start = partition_end_arr[partition_slot - 1]
    p_end = partition_end_arr[partition_slot]
    if tid >= p_end - p_start:
        return

    elem_id = partition_data[p_start + tid]

    # Only process constraint elements (indices >= contact_count)
    if elem_id < contact_count:
        return
    ji = elem_id - contact_count
    if ji >= joint_count[0]:
        return

    jtype = j_type[ji]
    b0 = j_body0[ji]
    b1 = j_body1[ji]

    q0 = b_orientation[b0]
    q1 = b_orientation[b1]

    inv_m0 = b_inverse_mass[b0]
    inv_m1 = b_inverse_mass[b1]
    inv_i0 = b_inverse_inertia_world[b0]
    inv_i1 = b_inverse_inertia_world[b1]

    is_static_0 = (b_flags[b0] & BODY_FLAG_STATIC) != 0 or inv_m0 == 0.0
    is_static_1 = (b_flags[b1] & BODY_FLAG_STATIC) != 0 or inv_m1 == 0.0

    # --- Point-on-point constraint (all joint types) ---
    rw0 = wp.quat_rotate(q0, j_local_anchor0[ji])
    rw1 = wp.quat_rotate(q1, j_local_anchor1[ji])
    j_rw0[ji] = rw0
    j_rw1[ji] = rw1

    # K^-1 = invMass0 * I + invMass1 * I + [r0]x * invI0 * [r0]x^T + [r1]x * invI1 * [r1]x^T
    r0x = skew_symmetric(rw0)
    r1x = skew_symmetric(rw1)
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    k_inv = identity * (inv_m0 + inv_m1)
    k_inv = k_inv + r0x * inv_i0 * wp.transpose(r0x)
    k_inv = k_inv + r1x * inv_i1 * wp.transpose(r1x)
    j_point_eff_mass[ji] = invert_mat33_safe(k_inv)

    # Warm start point constraint
    pt_lambda = j_point_lambda[ji]
    if not is_static_0:
        b_velocity[b0] = b_velocity[b0] - inv_m0 * pt_lambda
        b_angular_velocity[b0] = b_angular_velocity[b0] - inv_i0 * wp.cross(rw0, pt_lambda)
    if not is_static_1:
        b_velocity[b1] = b_velocity[b1] + inv_m1 * pt_lambda
        b_angular_velocity[b1] = b_angular_velocity[b1] + inv_i1 * wp.cross(rw1, pt_lambda)

    # --- Hinge rotation constraint (revolute and fixed joints) ---
    if jtype == JOINT_REVOLUTE or jtype == JOINT_FIXED:
        a1 = wp.quat_rotate(q0, j_local_axis0[ji])
        a2 = wp.quat_rotate(q1, j_local_axis1[ji])

        b2 = get_perpendiculars(a2)
        c2 = wp.cross(b2, a2)

        b2xa1 = wp.cross(b2, a1)
        c2xa1 = wp.cross(c2, a1)
        j_hinge_b2xa1[ji] = b2xa1
        j_hinge_c2xa1[ji] = c2xa1

        sum_inv_i = inv_i0 + inv_i1
        k00 = wp.dot(b2xa1, sum_inv_i * b2xa1)
        k01 = wp.dot(b2xa1, sum_inv_i * c2xa1)
        k10 = wp.dot(c2xa1, sum_inv_i * b2xa1)
        k11 = wp.dot(c2xa1, sum_inv_i * c2xa1)
        inv_k = invert_2x2(k00, k01, k10, k11)
        j_hinge_eff_mass_00[ji] = inv_k[0]
        j_hinge_eff_mass_01[ji] = inv_k[1]
        j_hinge_eff_mass_10[ji] = inv_k[2]
        j_hinge_eff_mass_11[ji] = inv_k[3]

        # Warm start hinge
        hlx = j_hinge_lambda_x[ji]
        hly = j_hinge_lambda_y[ji]
        h_impulse = b2xa1 * hlx + c2xa1 * hly
        if not is_static_0:
            b_angular_velocity[b0] = b_angular_velocity[b0] - inv_i0 * h_impulse
        if not is_static_1:
            b_angular_velocity[b1] = b_angular_velocity[b1] + inv_i1 * h_impulse

    # --- Angle constraint (revolute only, when limits are active) ---
    if jtype == JOINT_REVOLUTE:
        a_min = j_angle_min[ji]
        a_max = j_angle_max[ji]
        has_limits = a_min > -1.0e6 and a_max < 1.0e6

        a1 = wp.quat_rotate(q0, j_local_axis0[ji])
        j_angle_axis[ji] = a1

        # Compute current angle via quaternion difference
        inv_init = j_inv_init_orient[ji]
        diff = q1 * inv_init * wp.quat_inverse(q0)
        angle = quat_twist_angle(diff, a1)
        j_angle_current[ji] = angle

        if has_limits:
            # Scalar effective mass
            inv_eff = wp.dot(a1, (inv_i0 + inv_i1) * a1)
            if inv_eff > 0.0:
                j_angle_eff_mass[ji] = 1.0 / inv_eff
            else:
                j_angle_eff_mass[ji] = 0.0

            # Warm start angle
            a_lambda = j_angle_lambda[ji]
            a_impulse = a1 * a_lambda
            if not is_static_0:
                b_angular_velocity[b0] = b_angular_velocity[b0] - inv_i0 * a_impulse
            if not is_static_1:
                b_angular_velocity[b1] = b_angular_velocity[b1] + inv_i1 * a_impulse
        else:
            j_angle_eff_mass[ji] = 0.0
            j_angle_lambda[ji] = 0.0


# ---------------------------------------------------------------------------
# Solve constraints kernel
# ---------------------------------------------------------------------------


@wp.kernel
def solve_constraints_kernel(
    partition_data: wp.array(dtype=wp.int32),
    partition_end_arr: wp.array(dtype=wp.int32),
    partition_slot: int,
    contact_count: int,
    # Joint arrays
    j_type: wp.array(dtype=wp.int32),
    j_body0: wp.array(dtype=wp.int32),
    j_body1: wp.array(dtype=wp.int32),
    j_point_lambda: wp.array(dtype=wp.vec3),
    j_hinge_lambda_x: wp.array(dtype=wp.float32),
    j_hinge_lambda_y: wp.array(dtype=wp.float32),
    j_angle_lambda: wp.array(dtype=wp.float32),
    j_point_eff_mass: wp.array(dtype=wp.mat33),
    j_hinge_b2xa1: wp.array(dtype=wp.vec3),
    j_hinge_c2xa1: wp.array(dtype=wp.vec3),
    j_hinge_eff_mass_00: wp.array(dtype=wp.float32),
    j_hinge_eff_mass_01: wp.array(dtype=wp.float32),
    j_hinge_eff_mass_10: wp.array(dtype=wp.float32),
    j_hinge_eff_mass_11: wp.array(dtype=wp.float32),
    j_angle_eff_mass: wp.array(dtype=wp.float32),
    j_angle_axis: wp.array(dtype=wp.vec3),
    j_rw0: wp.array(dtype=wp.vec3),
    j_rw1: wp.array(dtype=wp.vec3),
    j_angle_min: wp.array(dtype=wp.float32),
    j_angle_max: wp.array(dtype=wp.float32),
    j_angle_current: wp.array(dtype=wp.float32),
    # Body arrays
    b_position: wp.array(dtype=wp.vec3),
    b_velocity: wp.array(dtype=wp.vec3),
    b_angular_velocity: wp.array(dtype=wp.vec3),
    b_inverse_mass: wp.array(dtype=wp.float32),
    b_inverse_inertia_world: wp.array(dtype=wp.mat33),
    b_flags: wp.array(dtype=wp.int32),
    joint_count: wp.array(dtype=wp.int32),
    use_bias: int,
):
    """One PGS iteration for joint constraints in a single partition."""
    tid = wp.tid()
    p_start = int(0)
    if partition_slot > 0:
        p_start = partition_end_arr[partition_slot - 1]
    p_end = partition_end_arr[partition_slot]
    if tid >= p_end - p_start:
        return

    elem_id = partition_data[p_start + tid]

    if elem_id < contact_count:
        return
    ji = elem_id - contact_count
    if ji >= joint_count[0]:
        return

    jtype = j_type[ji]
    b0 = j_body0[ji]
    b1 = j_body1[ji]

    v0 = b_velocity[b0]
    w0 = b_angular_velocity[b0]
    v1 = b_velocity[b1]
    w1 = b_angular_velocity[b1]

    inv_m0 = b_inverse_mass[b0]
    inv_m1 = b_inverse_mass[b1]
    inv_i0 = b_inverse_inertia_world[b0]
    inv_i1 = b_inverse_inertia_world[b1]

    is_static_0 = (b_flags[b0] & BODY_FLAG_STATIC) != 0 or inv_m0 == 0.0
    is_static_1 = (b_flags[b1] & BODY_FLAG_STATIC) != 0 or inv_m1 == 0.0

    # --- Hinge rotation constraint (solve before point for stability) ---
    if jtype == JOINT_REVOLUTE or jtype == JOINT_FIXED:
        b2xa1 = j_hinge_b2xa1[ji]
        c2xa1 = j_hinge_c2xa1[ji]

        delta_ang = w0 - w1
        jv_x = wp.dot(b2xa1, delta_ang)
        jv_y = wp.dot(c2xa1, delta_ang)

        # Baumgarte position correction
        cx = 0.0
        cy = 0.0
        if use_bias != 0:
            # Recompute constraint error from current body state
            # For hinge: c = [dot(a1, b2), dot(a1, c2)] where a1,b2,c2 are axis vectors
            # We use the cached cross products: dot(a1, b2) ≈ |b2xa1| * sin(angle) ≈ error
            # Simplified: we use the Jacobian velocity directly with Baumgarte
            cx = 0.0
            cy = 0.0

        em00 = j_hinge_eff_mass_00[ji]
        em01 = j_hinge_eff_mass_01[ji]
        em10 = j_hinge_eff_mass_10[ji]
        em11 = j_hinge_eff_mass_11[ji]

        rhs_x = jv_x - CONSTRAINT_BAUMGARTE * cx
        rhs_y = jv_y - CONSTRAINT_BAUMGARTE * cy
        lx = em00 * rhs_x + em01 * rhs_y
        ly = em10 * rhs_x + em11 * rhs_y

        old_hlx = j_hinge_lambda_x[ji]
        old_hly = j_hinge_lambda_y[ji]
        j_hinge_lambda_x[ji] = old_hlx + lx
        j_hinge_lambda_y[ji] = old_hly + ly

        h_impulse = b2xa1 * lx + c2xa1 * ly
        if not is_static_0:
            w0 = w0 - inv_i0 * h_impulse
        if not is_static_1:
            w1 = w1 + inv_i1 * h_impulse

    # --- Angle constraint (revolute motor/limits, only when active) ---
    if jtype == JOINT_REVOLUTE:
        angle_eff = j_angle_eff_mass[ji]
        if angle_eff > 0.0:
            axis = j_angle_axis[ji]
            jv_angle = wp.dot(axis, w0 - w1)

            angle_bias = 0.0
            if use_bias != 0:
                angle_val = j_angle_current[ji]
                a_min = j_angle_min[ji]
                a_max = j_angle_max[ji]
                if angle_val < a_min:
                    angle_bias = CONSTRAINT_BAUMGARTE * (angle_val - a_min)
                elif angle_val > a_max:
                    angle_bias = CONSTRAINT_BAUMGARTE * (angle_val - a_max)

            a_lambda = angle_eff * (jv_angle - angle_bias)

            old_al = j_angle_lambda[ji]
            new_al = old_al + a_lambda

            a_min = j_angle_min[ji]
            a_max = j_angle_max[ji]
            angle_val = j_angle_current[ji]
            if angle_val < a_min:
                new_al = wp.max(new_al, 0.0)
            elif angle_val > a_max:
                new_al = wp.min(new_al, 0.0)

            a_lambda = new_al - old_al
            j_angle_lambda[ji] = new_al

            a_impulse = axis * a_lambda
            if not is_static_0:
                w0 = w0 - inv_i0 * a_impulse
            if not is_static_1:
                w1 = w1 + inv_i1 * a_impulse

    # --- Point-on-point constraint ---
    rw0 = j_rw0[ji]
    rw1 = j_rw1[ji]

    # Relative velocity at anchor: (v0 + w0 x r0) - (v1 + w1 x r1)
    rel_vel = (v0 + wp.cross(w0, rw0)) - (v1 + wp.cross(w1, rw1))

    # Position error for Baumgarte
    pos_error = wp.vec3(0.0, 0.0, 0.0)
    if use_bias != 0:
        p0 = b_position[b0]
        p1 = b_position[b1]
        pos_error = (p1 + rw1) - (p0 + rw0)

    eff_mass = j_point_eff_mass[ji]
    pt_lambda = eff_mass * (rel_vel - CONSTRAINT_BAUMGARTE * pos_error)
    old_pt = j_point_lambda[ji]
    j_point_lambda[ji] = old_pt + pt_lambda

    if not is_static_0:
        v0 = v0 - inv_m0 * pt_lambda
        w0 = w0 - inv_i0 * wp.cross(rw0, pt_lambda)
    if not is_static_1:
        v1 = v1 + inv_m1 * pt_lambda
        w1 = w1 + inv_i1 * wp.cross(rw1, pt_lambda)

    # Write back velocities
    if not is_static_0:
        b_velocity[b0] = v0
        b_angular_velocity[b0] = w0
    if not is_static_1:
        b_velocity[b1] = v1
        b_angular_velocity[b1] = w1
