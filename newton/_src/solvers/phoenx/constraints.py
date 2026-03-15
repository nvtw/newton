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

Building blocks (composed inside :class:`ConstraintKernels`):
    - Point-on-point: 3-DOF position constraint (ball-socket anchor).
    - Hinge rotation: 2-DOF angular constraint (allows rotation around one axis).
    - Angle constraint: 1-DOF angular constraint (motor/limit on hinge axis).

Composite joints:
    - Ball-socket = PointOnPoint only.
    - Revolute = PointOnPoint + HingeRotation + AngleConstraint.
    - Fixed = PointOnPoint + HingeRotation.

Kernels receive only two flat ``float32`` arrays (the :class:`DataStore`
backing buffers for joints and bodies) plus partition metadata.  Column
offsets are baked at kernel-compile time via ``wp.static()``, and
``float_as_int`` / ``int_as_float`` (union-based reinterpret cast via
``wp.func_native``) handle integer columns stored in the float buffer.
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
    """Per-joint data stored in a :class:`DataStore`.

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
        hinge_lambda_x: Accumulated hinge impulse component 0 [N m s].
        hinge_lambda_y: Accumulated hinge impulse component 1 [N m s].
        angle_lambda: Accumulated impulse for angle constraint [N m s].
        point_eff_mass: 3x3 effective mass for point constraint.
        hinge_b2xa1: Cross product vector b2 x a1 (hinge).
        hinge_c2xa1: Cross product vector c2 x a1 (hinge).
        hinge_eff_mass_00: 2x2 effective mass element (0,0).
        hinge_eff_mass_01: 2x2 effective mass element (0,1).
        hinge_eff_mass_10: 2x2 effective mass element (1,0).
        hinge_eff_mass_11: 2x2 effective mass element (1,1).
        angle_eff_mass: Scalar effective mass for angle constraint.
        angle_axis: World-space angle constraint axis.
        rw0: World-space anchor offset for body 0 [m].
        rw1: World-space anchor offset for body 1 [m].
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
# Reinterpret cast (union-based, works on both CPU and CUDA)
# ---------------------------------------------------------------------------


@wp.func_native(snippet="union{float f;int i;}u;u.f=x;return u.i;")
def float_as_int(x: float) -> int:
    ...


@wp.func_native(snippet="union{float f;int i;}u;u.i=x;return u.f;")
def int_as_float(x: int) -> float:
    ...


# ---------------------------------------------------------------------------
# Typed accessors for flat float32 DataStore buffers
# ---------------------------------------------------------------------------


@wp.func
def ds_load_int(store: wp.array(dtype=wp.float32), base: int, row: int) -> int:
    return float_as_int(store[base + row])


@wp.func
def ds_store_int(store: wp.array(dtype=wp.float32), base: int, row: int, val: int):
    store[base + row] = int_as_float(val)


@wp.func
def ds_load_float(store: wp.array(dtype=wp.float32), base: int, row: int) -> float:
    return store[base + row]


@wp.func
def ds_store_float(store: wp.array(dtype=wp.float32), base: int, row: int, val: float):
    store[base + row] = val


@wp.func
def ds_load_vec3(store: wp.array(dtype=wp.float32), base: int, row: int) -> wp.vec3:
    i = base + row * 3
    return wp.vec3(store[i], store[i + 1], store[i + 2])


@wp.func
def ds_store_vec3(store: wp.array(dtype=wp.float32), base: int, row: int, v: wp.vec3):
    i = base + row * 3
    store[i] = v[0]
    store[i + 1] = v[1]
    store[i + 2] = v[2]


@wp.func
def ds_load_quat(store: wp.array(dtype=wp.float32), base: int, row: int) -> wp.quat:
    i = base + row * 4
    return wp.quat(store[i], store[i + 1], store[i + 2], store[i + 3])


@wp.func
def ds_load_mat33(store: wp.array(dtype=wp.float32), base: int, row: int) -> wp.mat33:
    i = base + row * 9
    return wp.mat33(
        store[i + 0], store[i + 1], store[i + 2],
        store[i + 3], store[i + 4], store[i + 5],
        store[i + 6], store[i + 7], store[i + 8],
    )


@wp.func
def ds_store_mat33(store: wp.array(dtype=wp.float32), base: int, row: int, m: wp.mat33):
    i = base + row * 9
    for r in range(3):
        for c in range(3):
            store[i + r * 3 + c] = m[r, c]


# ---------------------------------------------------------------------------
# Helper: compute column base index (flat offset into DataStore.data)
# ---------------------------------------------------------------------------


def col_base(store, name):
    """Return the flat float-index where column *name* starts.

    Suitable for ``wp.static()`` — the result is a plain ``int`` that
    depends only on the schema and capacity, not on the array pointer.
    """
    field = store.schema.fields[name]
    return field.col_offset * store.capacity


# ---------------------------------------------------------------------------
# Constraint math helpers
# ---------------------------------------------------------------------------


@wp.func
def skew_symmetric(v: wp.vec3) -> wp.mat33:
    return wp.mat33(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    )


@wp.func
def invert_mat33_safe(m: wp.mat33) -> wp.mat33:
    det = wp.determinant(m)
    if wp.abs(det) < 1.0e-12:
        return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return wp.inverse(m)


@wp.func
def invert_2x2(m00: float, m01: float, m10: float, m11: float) -> wp.vec4:
    det = m00 * m11 - m01 * m10
    if wp.abs(det) < 1.0e-12:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)
    inv_det = 1.0 / det
    return wp.vec4(m11 * inv_det, -m01 * inv_det, -m10 * inv_det, m00 * inv_det)


@wp.func
def get_perpendiculars(axis: wp.vec3) -> wp.vec3:
    if wp.abs(axis[0]) < 0.9:
        t = wp.cross(wp.vec3(1.0, 0.0, 0.0), axis)
    else:
        t = wp.cross(wp.vec3(0.0, 1.0, 0.0), axis)
    return wp.normalize(t)


@wp.func
def quat_twist_angle(q: wp.quat, axis: wp.vec3) -> float:
    qv = wp.vec3(q[0], q[1], q[2])
    proj = wp.dot(qv, axis)
    twist_len = wp.sqrt(proj * proj + q[3] * q[3])
    if twist_len < 1.0e-10:
        return 0.0
    angle = 2.0 * wp.atan2(wp.abs(proj), wp.abs(q[3]))
    if proj * q[3] < 0.0:
        angle = -angle
    return angle


# ---------------------------------------------------------------------------
# ConstraintKernels — bakes DataStore column offsets via wp.static()
# ---------------------------------------------------------------------------


class ConstraintKernels:
    """Compiled constraint kernels bound to specific joint and body stores.

    Each kernel receives only two flat ``float32`` arrays (the backing buffers
    of the joint and body :class:`DataStore` instances) plus partition metadata.
    Column offsets are baked as compile-time integer constants via
    ``wp.static(col_base(...))``.

    Args:
        joint_store: :class:`DataStore` for :class:`JointSchema`.
        body_store: :class:`HandleStore` for :class:`RigidBodySchema`.
    """

    def __init__(self, joint_store, body_store):
        body_ds = body_store.store  # HandleStore wraps a DataStore

        # Joint column base indices (baked via wp.static in kernels)
        j_type = col_base(joint_store, "joint_type")
        j_body0 = col_base(joint_store, "body0")
        j_body1 = col_base(joint_store, "body1")
        j_local_anchor0 = col_base(joint_store, "local_anchor0")
        j_local_anchor1 = col_base(joint_store, "local_anchor1")
        j_local_axis0 = col_base(joint_store, "local_axis0")
        j_local_axis1 = col_base(joint_store, "local_axis1")
        j_inv_init_orient = col_base(joint_store, "inv_initial_orientation")
        j_point_lambda = col_base(joint_store, "point_lambda")
        j_hinge_lambda_x = col_base(joint_store, "hinge_lambda_x")
        j_hinge_lambda_y = col_base(joint_store, "hinge_lambda_y")
        j_angle_lambda = col_base(joint_store, "angle_lambda")
        j_point_eff_mass = col_base(joint_store, "point_eff_mass")
        j_hinge_b2xa1 = col_base(joint_store, "hinge_b2xa1")
        j_hinge_c2xa1 = col_base(joint_store, "hinge_c2xa1")
        j_hinge_eff_mass_00 = col_base(joint_store, "hinge_eff_mass_00")
        j_hinge_eff_mass_01 = col_base(joint_store, "hinge_eff_mass_01")
        j_hinge_eff_mass_10 = col_base(joint_store, "hinge_eff_mass_10")
        j_hinge_eff_mass_11 = col_base(joint_store, "hinge_eff_mass_11")
        j_angle_eff_mass = col_base(joint_store, "angle_eff_mass")
        j_angle_axis = col_base(joint_store, "angle_axis")
        j_rw0 = col_base(joint_store, "rw0")
        j_rw1 = col_base(joint_store, "rw1")
        j_angle_min = col_base(joint_store, "angle_min")
        j_angle_max = col_base(joint_store, "angle_max")
        j_angle_current = col_base(joint_store, "angle_current")

        # Body column base indices
        b_position = col_base(body_ds, "position")
        b_orientation = col_base(body_ds, "orientation")
        b_velocity = col_base(body_ds, "velocity")
        b_angular_velocity = col_base(body_ds, "angular_velocity")
        b_inverse_mass = col_base(body_ds, "inverse_mass")
        b_inverse_inertia_world = col_base(body_ds, "inverse_inertia_world")
        b_flags = col_base(body_ds, "flags")

        # ---------------------------------------------------------------
        # Build elements kernel
        # ---------------------------------------------------------------

        @wp.kernel
        def _build_elements(
            jdata: wp.array(dtype=wp.float32),
            elements: wp.array2d(dtype=wp.int32),
            contact_count: wp.array(dtype=wp.int32),
            joint_count: wp.array(dtype=wp.int32),
        ):
            tid = wp.tid()
            if tid >= joint_count[0]:
                return
            row = contact_count[0] + tid
            elements[row, 0] = ds_load_int(jdata, wp.static(j_body0), tid)
            elements[row, 1] = ds_load_int(jdata, wp.static(j_body1), tid)
            for j in range(2, 8):
                elements[row, j] = -1

        # ---------------------------------------------------------------
        # Prepare kernel
        # ---------------------------------------------------------------

        @wp.kernel
        def _prepare(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            partition_data: wp.array(dtype=wp.int32),
            partition_end_arr: wp.array(dtype=wp.int32),
            partition_slot: int,
            contact_count: int,
            joint_count: wp.array(dtype=wp.int32),
        ):
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

            jtype = ds_load_int(jdata, wp.static(j_type), ji)
            b0 = ds_load_int(jdata, wp.static(j_body0), ji)
            b1 = ds_load_int(jdata, wp.static(j_body1), ji)

            q0 = ds_load_quat(bdata, wp.static(b_orientation), b0)
            q1 = ds_load_quat(bdata, wp.static(b_orientation), b1)

            inv_m0 = ds_load_float(bdata, wp.static(b_inverse_mass), b0)
            inv_m1 = ds_load_float(bdata, wp.static(b_inverse_mass), b1)
            inv_i0 = ds_load_mat33(bdata, wp.static(b_inverse_inertia_world), b0)
            inv_i1 = ds_load_mat33(bdata, wp.static(b_inverse_inertia_world), b1)

            f0 = ds_load_int(bdata, wp.static(b_flags), b0)
            f1 = ds_load_int(bdata, wp.static(b_flags), b1)
            is_static_0 = (f0 & BODY_FLAG_STATIC) != 0 or inv_m0 == 0.0
            is_static_1 = (f1 & BODY_FLAG_STATIC) != 0 or inv_m1 == 0.0

            # --- Point-on-point constraint (all joint types) ---
            rw0 = wp.quat_rotate(q0, ds_load_vec3(jdata, wp.static(j_local_anchor0), ji))
            rw1 = wp.quat_rotate(q1, ds_load_vec3(jdata, wp.static(j_local_anchor1), ji))
            ds_store_vec3(jdata, wp.static(j_rw0), ji, rw0)
            ds_store_vec3(jdata, wp.static(j_rw1), ji, rw1)

            r0x = skew_symmetric(rw0)
            r1x = skew_symmetric(rw1)
            identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            k_inv = identity * (inv_m0 + inv_m1)
            k_inv = k_inv + r0x * inv_i0 * wp.transpose(r0x)
            k_inv = k_inv + r1x * inv_i1 * wp.transpose(r1x)
            ds_store_mat33(jdata, wp.static(j_point_eff_mass), ji, invert_mat33_safe(k_inv))

            # Warm start point
            pt_l = ds_load_vec3(jdata, wp.static(j_point_lambda), ji)
            if not is_static_0:
                v0 = ds_load_vec3(bdata, wp.static(b_velocity), b0)
                w0 = ds_load_vec3(bdata, wp.static(b_angular_velocity), b0)
                ds_store_vec3(bdata, wp.static(b_velocity), b0, v0 - inv_m0 * pt_l)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b0, w0 - inv_i0 * wp.cross(rw0, pt_l))
            if not is_static_1:
                v1 = ds_load_vec3(bdata, wp.static(b_velocity), b1)
                w1 = ds_load_vec3(bdata, wp.static(b_angular_velocity), b1)
                ds_store_vec3(bdata, wp.static(b_velocity), b1, v1 + inv_m1 * pt_l)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b1, w1 + inv_i1 * wp.cross(rw1, pt_l))

            # --- Hinge rotation constraint (revolute and fixed) ---
            if jtype == JOINT_REVOLUTE or jtype == JOINT_FIXED:
                a1 = wp.quat_rotate(q0, ds_load_vec3(jdata, wp.static(j_local_axis0), ji))
                a2 = wp.quat_rotate(q1, ds_load_vec3(jdata, wp.static(j_local_axis1), ji))

                b2 = get_perpendiculars(a2)
                c2 = wp.cross(b2, a2)
                b2xa1 = wp.cross(b2, a1)
                c2xa1 = wp.cross(c2, a1)
                ds_store_vec3(jdata, wp.static(j_hinge_b2xa1), ji, b2xa1)
                ds_store_vec3(jdata, wp.static(j_hinge_c2xa1), ji, c2xa1)

                sum_inv_i = inv_i0 + inv_i1
                k00 = wp.dot(b2xa1, sum_inv_i * b2xa1)
                k01 = wp.dot(b2xa1, sum_inv_i * c2xa1)
                k10 = wp.dot(c2xa1, sum_inv_i * b2xa1)
                k11 = wp.dot(c2xa1, sum_inv_i * c2xa1)
                inv_k = invert_2x2(k00, k01, k10, k11)
                ds_store_float(jdata, wp.static(j_hinge_eff_mass_00), ji, inv_k[0])
                ds_store_float(jdata, wp.static(j_hinge_eff_mass_01), ji, inv_k[1])
                ds_store_float(jdata, wp.static(j_hinge_eff_mass_10), ji, inv_k[2])
                ds_store_float(jdata, wp.static(j_hinge_eff_mass_11), ji, inv_k[3])

                hlx = ds_load_float(jdata, wp.static(j_hinge_lambda_x), ji)
                hly = ds_load_float(jdata, wp.static(j_hinge_lambda_y), ji)
                h_imp = b2xa1 * hlx + c2xa1 * hly
                if not is_static_0:
                    w0_h = ds_load_vec3(bdata, wp.static(b_angular_velocity), b0)
                    ds_store_vec3(bdata, wp.static(b_angular_velocity), b0, w0_h - inv_i0 * h_imp)
                if not is_static_1:
                    w1_h = ds_load_vec3(bdata, wp.static(b_angular_velocity), b1)
                    ds_store_vec3(bdata, wp.static(b_angular_velocity), b1, w1_h + inv_i1 * h_imp)

            # --- Angle constraint (revolute only, when limits are active) ---
            if jtype == JOINT_REVOLUTE:
                a_min_val = ds_load_float(jdata, wp.static(j_angle_min), ji)
                a_max_val = ds_load_float(jdata, wp.static(j_angle_max), ji)
                has_limits = a_min_val > -1.0e6 and a_max_val < 1.0e6

                a1 = wp.quat_rotate(q0, ds_load_vec3(jdata, wp.static(j_local_axis0), ji))
                ds_store_vec3(jdata, wp.static(j_angle_axis), ji, a1)

                inv_init = ds_load_quat(jdata, wp.static(j_inv_init_orient), ji)
                diff = q1 * inv_init * wp.quat_inverse(q0)
                angle = quat_twist_angle(diff, a1)
                ds_store_float(jdata, wp.static(j_angle_current), ji, angle)

                if has_limits:
                    inv_eff = wp.dot(a1, (inv_i0 + inv_i1) * a1)
                    if inv_eff > 0.0:
                        ds_store_float(jdata, wp.static(j_angle_eff_mass), ji, 1.0 / inv_eff)
                    else:
                        ds_store_float(jdata, wp.static(j_angle_eff_mass), ji, 0.0)

                    a_l = ds_load_float(jdata, wp.static(j_angle_lambda), ji)
                    a_imp = a1 * a_l
                    if not is_static_0:
                        w0_a = ds_load_vec3(bdata, wp.static(b_angular_velocity), b0)
                        ds_store_vec3(bdata, wp.static(b_angular_velocity), b0, w0_a - inv_i0 * a_imp)
                    if not is_static_1:
                        w1_a = ds_load_vec3(bdata, wp.static(b_angular_velocity), b1)
                        ds_store_vec3(bdata, wp.static(b_angular_velocity), b1, w1_a + inv_i1 * a_imp)
                else:
                    ds_store_float(jdata, wp.static(j_angle_eff_mass), ji, 0.0)
                    ds_store_float(jdata, wp.static(j_angle_lambda), ji, 0.0)

        # ---------------------------------------------------------------
        # Solve kernel
        # ---------------------------------------------------------------

        @wp.kernel
        def _solve(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            partition_data: wp.array(dtype=wp.int32),
            partition_end_arr: wp.array(dtype=wp.int32),
            partition_slot: int,
            contact_count: int,
            joint_count: wp.array(dtype=wp.int32),
            use_bias: int,
        ):
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

            jtype = ds_load_int(jdata, wp.static(j_type), ji)
            b0 = ds_load_int(jdata, wp.static(j_body0), ji)
            b1 = ds_load_int(jdata, wp.static(j_body1), ji)

            v0 = ds_load_vec3(bdata, wp.static(b_velocity), b0)
            w0 = ds_load_vec3(bdata, wp.static(b_angular_velocity), b0)
            v1 = ds_load_vec3(bdata, wp.static(b_velocity), b1)
            w1 = ds_load_vec3(bdata, wp.static(b_angular_velocity), b1)

            inv_m0 = ds_load_float(bdata, wp.static(b_inverse_mass), b0)
            inv_m1 = ds_load_float(bdata, wp.static(b_inverse_mass), b1)
            inv_i0 = ds_load_mat33(bdata, wp.static(b_inverse_inertia_world), b0)
            inv_i1 = ds_load_mat33(bdata, wp.static(b_inverse_inertia_world), b1)

            f0 = ds_load_int(bdata, wp.static(b_flags), b0)
            f1 = ds_load_int(bdata, wp.static(b_flags), b1)
            is_static_0 = (f0 & BODY_FLAG_STATIC) != 0 or inv_m0 == 0.0
            is_static_1 = (f1 & BODY_FLAG_STATIC) != 0 or inv_m1 == 0.0

            # --- Hinge rotation ---
            if jtype == JOINT_REVOLUTE or jtype == JOINT_FIXED:
                b2xa1 = ds_load_vec3(jdata, wp.static(j_hinge_b2xa1), ji)
                c2xa1 = ds_load_vec3(jdata, wp.static(j_hinge_c2xa1), ji)

                delta_ang = w0 - w1
                jv_x = wp.dot(b2xa1, delta_ang)
                jv_y = wp.dot(c2xa1, delta_ang)

                em00 = ds_load_float(jdata, wp.static(j_hinge_eff_mass_00), ji)
                em01 = ds_load_float(jdata, wp.static(j_hinge_eff_mass_01), ji)
                em10 = ds_load_float(jdata, wp.static(j_hinge_eff_mass_10), ji)
                em11 = ds_load_float(jdata, wp.static(j_hinge_eff_mass_11), ji)

                lx = em00 * jv_x + em01 * jv_y
                ly = em10 * jv_x + em11 * jv_y

                ds_store_float(jdata, wp.static(j_hinge_lambda_x), ji,
                               ds_load_float(jdata, wp.static(j_hinge_lambda_x), ji) + lx)
                ds_store_float(jdata, wp.static(j_hinge_lambda_y), ji,
                               ds_load_float(jdata, wp.static(j_hinge_lambda_y), ji) + ly)

                h_imp = b2xa1 * lx + c2xa1 * ly
                if not is_static_0:
                    w0 = w0 - inv_i0 * h_imp
                if not is_static_1:
                    w1 = w1 + inv_i1 * h_imp

            # --- Angle constraint (only when active) ---
            if jtype == JOINT_REVOLUTE:
                angle_eff = ds_load_float(jdata, wp.static(j_angle_eff_mass), ji)
                if angle_eff > 0.0:
                    axis = ds_load_vec3(jdata, wp.static(j_angle_axis), ji)
                    jv_a = wp.dot(axis, w0 - w1)

                    angle_bias = 0.0
                    if use_bias != 0:
                        angle_val = ds_load_float(jdata, wp.static(j_angle_current), ji)
                        a_min_v = ds_load_float(jdata, wp.static(j_angle_min), ji)
                        a_max_v = ds_load_float(jdata, wp.static(j_angle_max), ji)
                        if angle_val < a_min_v:
                            angle_bias = CONSTRAINT_BAUMGARTE * (angle_val - a_min_v)
                        elif angle_val > a_max_v:
                            angle_bias = CONSTRAINT_BAUMGARTE * (angle_val - a_max_v)

                    a_l = angle_eff * (jv_a - angle_bias)
                    old_al = ds_load_float(jdata, wp.static(j_angle_lambda), ji)
                    new_al = old_al + a_l

                    a_min_v = ds_load_float(jdata, wp.static(j_angle_min), ji)
                    a_max_v = ds_load_float(jdata, wp.static(j_angle_max), ji)
                    angle_val = ds_load_float(jdata, wp.static(j_angle_current), ji)
                    if angle_val < a_min_v:
                        new_al = wp.max(new_al, 0.0)
                    elif angle_val > a_max_v:
                        new_al = wp.min(new_al, 0.0)

                    a_l = new_al - old_al
                    ds_store_float(jdata, wp.static(j_angle_lambda), ji, new_al)

                    a_imp = axis * a_l
                    if not is_static_0:
                        w0 = w0 - inv_i0 * a_imp
                    if not is_static_1:
                        w1 = w1 + inv_i1 * a_imp

            # --- Point-on-point ---
            rw0 = ds_load_vec3(jdata, wp.static(j_rw0), ji)
            rw1 = ds_load_vec3(jdata, wp.static(j_rw1), ji)

            rel_vel = (v0 + wp.cross(w0, rw0)) - (v1 + wp.cross(w1, rw1))

            pos_error = wp.vec3(0.0, 0.0, 0.0)
            if use_bias != 0:
                p0 = ds_load_vec3(bdata, wp.static(b_position), b0)
                p1 = ds_load_vec3(bdata, wp.static(b_position), b1)
                pos_error = (p1 + rw1) - (p0 + rw0)

            eff = ds_load_mat33(jdata, wp.static(j_point_eff_mass), ji)
            pt_l = eff * (rel_vel - CONSTRAINT_BAUMGARTE * pos_error)
            old_pt = ds_load_vec3(jdata, wp.static(j_point_lambda), ji)
            ds_store_vec3(jdata, wp.static(j_point_lambda), ji, old_pt + pt_l)

            if not is_static_0:
                v0 = v0 - inv_m0 * pt_l
                w0 = w0 - inv_i0 * wp.cross(rw0, pt_l)
            if not is_static_1:
                v1 = v1 + inv_m1 * pt_l
                w1 = w1 + inv_i1 * wp.cross(rw1, pt_l)

            # Write back
            if not is_static_0:
                ds_store_vec3(bdata, wp.static(b_velocity), b0, v0)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b0, w0)
            if not is_static_1:
                ds_store_vec3(bdata, wp.static(b_velocity), b1, v1)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b1, w1)

        # Store compiled kernels and data arrays for launch
        self.build_elements = _build_elements
        self.prepare = _prepare
        self.solve = _solve
        self.joint_data = joint_store.data
        self.body_data = body_ds.data
        self.joint_count = joint_store.count
        self.device = joint_store.device
        self.joint_capacity = joint_store.capacity
