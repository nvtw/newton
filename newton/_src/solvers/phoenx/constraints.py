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

Each joint type has its own ``@wp.struct`` schema defining its per-row
data layout.  All schemas share a common header (body indices, anchors,
axes, orientation) so that common fields sit at identical flat-buffer
offsets.  The joint :class:`~newton._src.solvers.phoenx.data_base.DataStore`
is sized to ``max(floats_per_row)`` across all types — different joint
types **overlay** the same physical memory after the common header,
exactly like a C union.

Building blocks (composed inside :class:`ConstraintKernels`):
    - Point-on-point: 3-DOF position constraint (ball-socket anchor).
    - Hinge rotation: 2-DOF angular constraint (allows rotation around one axis).
    - Angle constraint: 1-DOF angular constraint (motor/limit on hinge axis).
    - Slide constraint: 1-DOF linear constraint (prismatic axis).

Composite joints:
    - Ball-socket = PointOnPoint.
    - Revolute = PointOnPoint + HingeRotation + AngleConstraint + Drive.
    - Fixed = PointOnPoint + HingeRotation.
    - Prismatic = SlidePerp + HingeRotation + SlideAxis + Drive.
"""

import warp as wp

from .data_base import Schema
from .schemas import BODY_FLAG_STATIC

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSTRAINT_BAUMGARTE = wp.constant(0.2)

# Constraint type IDs
JOINT_BALL_SOCKET = 0
JOINT_REVOLUTE = 1
JOINT_FIXED = 2
JOINT_PRISMATIC = 3
JOINT_DISTANCE_LIMIT = 4

# Drive mode constants
DRIVE_OFF = 0
DRIVE_POSITION = 1
DRIVE_VELOCITY = 2

# Constraint access mode (matches C# ConstraintAccessMode)
ACCESS_MODE_VELOCITY_LEVEL = 0
ACCESS_MODE_POSITION_LEVEL = 1

# ---------------------------------------------------------------------------
# Per-type joint schemas (union-style: overlapping storage)
#
# IMPORTANT: the first 10 fields (the "common header") MUST appear in
# the same order and with the same types in every schema so that their
# flat-buffer offsets are identical regardless of joint type.  Fields
# after the header are type-specific and overlay the same memory.
# ---------------------------------------------------------------------------


@wp.struct
class BallSocketJointData:
    """Ball-socket joint: 3-DOF point constraint only."""

    # --- common header (offset 0–24, shared by all types) ---
    joint_type: wp.int32
    body0: wp.int32
    body1: wp.int32
    local_anchor0: wp.vec3
    local_anchor1: wp.vec3
    local_axis0: wp.vec3
    local_axis1: wp.vec3
    inv_initial_orientation: wp.quat
    rw0: wp.vec3
    rw1: wp.vec3
    # --- point constraint ---
    point_lambda: wp.vec3
    point_eff_mass: wp.mat33


@wp.struct
class FixedJointData:
    """Fixed (weld) joint: point constraint + full rotation lock."""

    # --- common header ---
    joint_type: wp.int32
    body0: wp.int32
    body1: wp.int32
    local_anchor0: wp.vec3
    local_anchor1: wp.vec3
    local_axis0: wp.vec3
    local_axis1: wp.vec3
    inv_initial_orientation: wp.quat
    rw0: wp.vec3
    rw1: wp.vec3
    # --- point constraint (overlaps BallSocket at offset 25) ---
    point_lambda: wp.vec3
    point_eff_mass: wp.mat33
    # --- hinge rotation lock (2-DOF angular) ---
    hinge_lambda_x: wp.float32
    hinge_lambda_y: wp.float32
    hinge_b2xa1: wp.vec3
    hinge_c2xa1: wp.vec3
    hinge_eff_mass_00: wp.float32
    hinge_eff_mass_01: wp.float32
    hinge_eff_mass_10: wp.float32
    hinge_eff_mass_11: wp.float32


@wp.struct
class RevoluteJointData:
    """Revolute (hinge) joint: point + rotation lock + angle limits + drive."""

    # --- common header ---
    joint_type: wp.int32
    body0: wp.int32
    body1: wp.int32
    local_anchor0: wp.vec3
    local_anchor1: wp.vec3
    local_axis0: wp.vec3
    local_axis1: wp.vec3
    inv_initial_orientation: wp.quat
    rw0: wp.vec3
    rw1: wp.vec3
    # --- point constraint (offset 25) ---
    point_lambda: wp.vec3
    point_eff_mass: wp.mat33
    # --- hinge rotation (offset 37) ---
    hinge_lambda_x: wp.float32
    hinge_lambda_y: wp.float32
    hinge_b2xa1: wp.vec3
    hinge_c2xa1: wp.vec3
    hinge_eff_mass_00: wp.float32
    hinge_eff_mass_01: wp.float32
    hinge_eff_mass_10: wp.float32
    hinge_eff_mass_11: wp.float32
    # --- angle constraint (offset 49) ---
    angle_lambda: wp.float32
    angle_eff_mass: wp.float32
    angle_axis: wp.vec3
    angle_current: wp.float32
    angle_min: wp.float32
    angle_max: wp.float32
    # --- drive (offset 57) ---
    drive_mode: wp.int32
    drive_target: wp.float32
    drive_stiffness: wp.float32
    drive_damping: wp.float32
    drive_max_force: wp.float32
    drive_lambda: wp.float32
    drive_eff_mass: wp.float32


@wp.struct
class PrismaticJointData:
    """Prismatic (slider) joint: slide perp + rotation lock + slide axis + drive."""

    # --- common header ---
    joint_type: wp.int32
    body0: wp.int32
    body1: wp.int32
    local_anchor0: wp.vec3
    local_anchor1: wp.vec3
    local_axis0: wp.vec3
    local_axis1: wp.vec3
    inv_initial_orientation: wp.quat
    rw0: wp.vec3
    rw1: wp.vec3
    # --- slide perpendicular constraint (offset 25, overlaps point in revolute) ---
    slide_perp_lambda_x: wp.float32
    slide_perp_lambda_y: wp.float32
    slide_perp0: wp.vec3
    slide_perp1: wp.vec3
    slide_perp_eff_00: wp.float32
    slide_perp_eff_01: wp.float32
    slide_perp_eff_10: wp.float32
    slide_perp_eff_11: wp.float32
    # --- rotation lock (offset 37, same as hinge in revolute/fixed) ---
    hinge_lambda_x: wp.float32
    hinge_lambda_y: wp.float32
    hinge_b2xa1: wp.vec3
    hinge_c2xa1: wp.vec3
    hinge_eff_mass_00: wp.float32
    hinge_eff_mass_01: wp.float32
    hinge_eff_mass_10: wp.float32
    hinge_eff_mass_11: wp.float32
    # --- slide axis constraint (offset 49, overlaps angle in revolute) ---
    slide_lambda: wp.float32
    slide_eff_mass: wp.float32
    slide_axis: wp.vec3
    slide_current: wp.float32
    slide_min: wp.float32
    slide_max: wp.float32
    # --- drive (offset 57, same as drive in revolute) ---
    drive_mode: wp.int32
    drive_target: wp.float32
    drive_stiffness: wp.float32
    drive_damping: wp.float32
    drive_max_force: wp.float32
    drive_lambda: wp.float32
    drive_eff_mass: wp.float32


@wp.struct
class DistanceLimitJointData:
    """Distance limit with optional spring: 1-DOF along distance axis."""

    # --- common header (offset 0-24, shared by all types) ---
    joint_type: wp.int32
    body0: wp.int32
    body1: wp.int32
    local_anchor0: wp.vec3
    local_anchor1: wp.vec3
    local_axis0: wp.vec3
    local_axis1: wp.vec3
    inv_initial_orientation: wp.quat
    rw0: wp.vec3
    rw1: wp.vec3
    # --- distance limit specific ---
    rest_distance: wp.float32
    limit_min: wp.float32
    limit_max: wp.float32
    softness: wp.float32
    bias_factor: wp.float32
    dl_eff_mass: wp.float32
    dl_accumulated_impulse: wp.float32
    dl_bias: wp.float32
    dl_j0: wp.vec3
    dl_j1: wp.vec3
    dl_j2: wp.vec3
    dl_j3: wp.vec3
    dl_clamp: wp.int32


# All joint types, used to compute max floats_per_row.
ALL_JOINT_TYPES = (BallSocketJointData, FixedJointData, RevoluteJointData, PrismaticJointData, DistanceLimitJointData)

# Backwards-compatible alias used for DataStore creation and host-side column_of.
# This is the largest schema (RevoluteJointData == PrismaticJointData == 64 floats).
JointSchema = RevoluteJointData


def joint_max_floats_per_row() -> int:
    """Return the maximum floats-per-row across all joint schemas."""
    return max(Schema(t).floats_per_row for t in ALL_JOINT_TYPES)


# ---------------------------------------------------------------------------
# Reinterpret cast (union-based, works on both CPU and CUDA)
# ---------------------------------------------------------------------------


@wp.func_native(snippet="union{float f;int i;}u;u.f=x;return u.i;")
def float_as_int(x: float) -> int: ...


@wp.func_native(snippet="union{float f;int i;}u;u.i=x;return u.f;")
def int_as_float(x: int) -> float: ...


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
        store[i + 0],
        store[i + 1],
        store[i + 2],
        store[i + 3],
        store[i + 4],
        store[i + 5],
        store[i + 6],
        store[i + 7],
        store[i + 8],
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

_schema_cache: dict[type, Schema] = {}


def _get_schema(struct_type) -> Schema:
    """Return (cached) Schema for a ``@wp.struct`` type."""
    if struct_type not in _schema_cache:
        _schema_cache[struct_type] = Schema(struct_type)
    return _schema_cache[struct_type]


def col_base(store, name):
    """Return the flat float-index where column *name* starts.

    Suitable for ``wp.static()`` — the result is a plain ``int`` that
    depends only on the schema and capacity, not on the array pointer.
    """
    field = store.schema.fields[name]
    return field.col_offset * store.capacity


def schema_col_base(struct_type, capacity: int, name: str) -> int:
    """Compute col_base for a field from a per-type schema.

    Unlike :func:`col_base` which requires a DataStore, this works with
    any ``@wp.struct`` type and a capacity.  Use this when computing
    offsets for union-style joint schemas where different types overlay
    the same memory.
    """
    schema = _get_schema(struct_type)
    field = schema.fields[name]
    return field.col_offset * capacity


def schema_column_of(data, struct_type, capacity: int, name: str, device):
    """Get a typed wp.array view for a field from a per-type schema.

    Like :meth:`DataStore.column_of` but resolves the field from the
    given *struct_type* schema, allowing access to union-overlapping fields.
    """
    schema = _get_schema(struct_type)
    field = schema.fields[name]
    byte_offset = field.col_offset * capacity * 4
    return wp.array(
        ptr=data.ptr + byte_offset,
        dtype=field.dtype,
        shape=(capacity,),
        device=device,
        copy=False,
    )


# ---------------------------------------------------------------------------
# Constraint math helpers
# ---------------------------------------------------------------------------


@wp.func
def skew_symmetric(v: wp.vec3) -> wp.mat33:
    return wp.mat33(
        0.0,
        -v[2],
        v[1],
        v[2],
        0.0,
        -v[0],
        -v[1],
        v[0],
        0.0,
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
# Constraint access mode: velocity ↔ position state conversion
#
# Ported from C# TinyRigidState.SynchronizeVelAndPosStateUpdates.
# When switching a body between velocity-level and position-level
# solving, these functions convert the accumulated delta from one
# representation to the other.
# ---------------------------------------------------------------------------


@wp.func
def sync_pos_to_vel(
    ref_pos: wp.vec3,
    ref_orient: wp.quat,
    cur_pos: wp.vec3,
    cur_orient: wp.quat,
    inv_dt: float,
):
    """Convert a position-level state delta back to velocity.

    Given a reference (start-of-step) position/orientation and the
    current (modified by position-level constraints) position/orientation,
    compute the equivalent linear and angular velocity.

    Matches C# Algorithm 2 from "Detailed Rigid Body Simulation with
    Extended Position Based Dynamics".

    Returns:
        vel: linear velocity [m/s].
        angvel: angular velocity [rad/s].
    """
    vel = (cur_pos - ref_pos) * inv_dt

    delta_q = cur_orient * wp.quat_inverse(ref_orient)
    qv = wp.vec3(delta_q[0], delta_q[1], delta_q[2])
    angvel = 2.0 * inv_dt * qv
    if delta_q[3] < 0.0:
        angvel = -angvel

    return vel, angvel


@wp.func
def sync_vel_to_pos(
    ref_pos: wp.vec3,
    ref_orient: wp.quat,
    vel: wp.vec3,
    angvel: wp.vec3,
    dt: float,
):
    """Convert velocity-level state to a position-level state.

    Integrates the given velocity forward by *dt* from the reference
    state to produce a new position and orientation.

    Returns:
        new_pos: integrated position [m].
        new_orient: integrated orientation (unit quaternion).
    """
    new_pos = ref_pos + vel * dt

    # Quaternion integration: q' = q + 0.5 * dt * [wx, wy, wz, 0] * q
    half_dt = 0.5 * dt
    dq = wp.quat(
        half_dt * (angvel[0] * ref_orient[3] + angvel[1] * ref_orient[2] - angvel[2] * ref_orient[1]),
        half_dt * (-angvel[0] * ref_orient[2] + angvel[1] * ref_orient[3] + angvel[2] * ref_orient[0]),
        half_dt * (angvel[0] * ref_orient[1] - angvel[1] * ref_orient[0] + angvel[2] * ref_orient[3]),
        half_dt * (-angvel[0] * ref_orient[0] - angvel[1] * ref_orient[1] - angvel[2] * ref_orient[2]),
    )
    new_orient = wp.normalize(
        wp.quat(
            ref_orient[0] + dq[0],
            ref_orient[1] + dq[1],
            ref_orient[2] + dq[2],
            ref_orient[3] + dq[3],
        )
    )

    return new_pos, new_orient


# ---------------------------------------------------------------------------
# ConstraintKernels — bakes DataStore column offsets via wp.static()
# ---------------------------------------------------------------------------


class ConstraintKernels:
    """Compiled constraint kernels bound to specific joint and body stores.

    Column offsets are computed from **per-type schemas** so that
    different joint types correctly overlay the same physical memory
    (union-style).  The common header fields (body0, body1, anchors,
    axes) are at identical offsets across all types.

    Each joint type has its own ``wp.func`` for prepare and solve,
    following the C# PhoenX pattern where each joint composes
    reusable building-block constraint parts.

    Args:
        joint_store: :class:`DataStore` for joint data (sized to
            :func:`joint_max_floats_per_row`).
        body_store: :class:`HandleStore` for :class:`RigidBodySchema`.
    """

    def __init__(self, joint_store, body_store):
        body_ds = body_store.store  # HandleStore wraps a DataStore
        cap = joint_store.capacity

        # -- Common header offsets (identical across all schemas) ----------
        j_type = schema_col_base(BallSocketJointData, cap, "joint_type")
        j_body0 = schema_col_base(BallSocketJointData, cap, "body0")
        j_body1 = schema_col_base(BallSocketJointData, cap, "body1")
        j_local_anchor0 = schema_col_base(BallSocketJointData, cap, "local_anchor0")
        j_local_anchor1 = schema_col_base(BallSocketJointData, cap, "local_anchor1")
        j_local_axis0 = schema_col_base(BallSocketJointData, cap, "local_axis0")
        j_local_axis1 = schema_col_base(BallSocketJointData, cap, "local_axis1")
        j_inv_init_orient = schema_col_base(BallSocketJointData, cap, "inv_initial_orientation")
        j_rw0 = schema_col_base(BallSocketJointData, cap, "rw0")
        j_rw1 = schema_col_base(BallSocketJointData, cap, "rw1")

        # -- Point constraint offsets (BallSocket/Revolute/Fixed) ---------
        j_point_lambda = schema_col_base(BallSocketJointData, cap, "point_lambda")
        j_point_eff_mass = schema_col_base(BallSocketJointData, cap, "point_eff_mass")

        # -- Hinge rotation offsets (Revolute/Fixed/Prismatic — same) -----
        j_hinge_lambda_x = schema_col_base(RevoluteJointData, cap, "hinge_lambda_x")
        j_hinge_lambda_y = schema_col_base(RevoluteJointData, cap, "hinge_lambda_y")
        j_hinge_b2xa1 = schema_col_base(RevoluteJointData, cap, "hinge_b2xa1")
        j_hinge_c2xa1 = schema_col_base(RevoluteJointData, cap, "hinge_c2xa1")
        j_hinge_eff_mass_00 = schema_col_base(RevoluteJointData, cap, "hinge_eff_mass_00")
        j_hinge_eff_mass_01 = schema_col_base(RevoluteJointData, cap, "hinge_eff_mass_01")
        j_hinge_eff_mass_10 = schema_col_base(RevoluteJointData, cap, "hinge_eff_mass_10")
        j_hinge_eff_mass_11 = schema_col_base(RevoluteJointData, cap, "hinge_eff_mass_11")

        # -- Angle constraint offsets (Revolute only) ---------------------
        j_angle_lambda = schema_col_base(RevoluteJointData, cap, "angle_lambda")
        j_angle_eff_mass = schema_col_base(RevoluteJointData, cap, "angle_eff_mass")
        j_angle_axis = schema_col_base(RevoluteJointData, cap, "angle_axis")
        j_angle_current = schema_col_base(RevoluteJointData, cap, "angle_current")
        j_angle_min = schema_col_base(RevoluteJointData, cap, "angle_min")
        j_angle_max = schema_col_base(RevoluteJointData, cap, "angle_max")

        # -- Slide perpendicular offsets (Prismatic, overlaps point) ------
        j_slide_perp_lambda_x = schema_col_base(PrismaticJointData, cap, "slide_perp_lambda_x")
        j_slide_perp_lambda_y = schema_col_base(PrismaticJointData, cap, "slide_perp_lambda_y")
        j_slide_perp0 = schema_col_base(PrismaticJointData, cap, "slide_perp0")
        j_slide_perp1 = schema_col_base(PrismaticJointData, cap, "slide_perp1")
        j_slide_perp_eff_00 = schema_col_base(PrismaticJointData, cap, "slide_perp_eff_00")
        j_slide_perp_eff_01 = schema_col_base(PrismaticJointData, cap, "slide_perp_eff_01")
        j_slide_perp_eff_10 = schema_col_base(PrismaticJointData, cap, "slide_perp_eff_10")
        j_slide_perp_eff_11 = schema_col_base(PrismaticJointData, cap, "slide_perp_eff_11")

        # -- Slide axis offsets (Prismatic, overlaps angle in Revolute) ---
        j_slide_lambda = schema_col_base(PrismaticJointData, cap, "slide_lambda")
        j_slide_eff_mass = schema_col_base(PrismaticJointData, cap, "slide_eff_mass")
        j_slide_axis = schema_col_base(PrismaticJointData, cap, "slide_axis")
        j_slide_current = schema_col_base(PrismaticJointData, cap, "slide_current")
        j_slide_min = schema_col_base(PrismaticJointData, cap, "slide_min")
        j_slide_max = schema_col_base(PrismaticJointData, cap, "slide_max")

        # -- Drive offsets (Revolute/Prismatic — same offset) -------------
        j_drive_mode = schema_col_base(RevoluteJointData, cap, "drive_mode")
        j_drive_target = schema_col_base(RevoluteJointData, cap, "drive_target")
        j_drive_stiffness = schema_col_base(RevoluteJointData, cap, "drive_stiffness")
        j_drive_damping = schema_col_base(RevoluteJointData, cap, "drive_damping")
        j_drive_max_force = schema_col_base(RevoluteJointData, cap, "drive_max_force")
        j_drive_lambda = schema_col_base(RevoluteJointData, cap, "drive_lambda")
        j_drive_eff_mass = schema_col_base(RevoluteJointData, cap, "drive_eff_mass")

        # -- Distance limit column offsets --------------------------------
        dl_rest_distance = schema_col_base(DistanceLimitJointData, cap, "rest_distance")
        dl_limit_min = schema_col_base(DistanceLimitJointData, cap, "limit_min")
        dl_limit_max = schema_col_base(DistanceLimitJointData, cap, "limit_max")
        dl_softness = schema_col_base(DistanceLimitJointData, cap, "softness")
        dl_bias_factor = schema_col_base(DistanceLimitJointData, cap, "bias_factor")
        dl_eff_mass = schema_col_base(DistanceLimitJointData, cap, "dl_eff_mass")
        dl_accumulated_impulse = schema_col_base(DistanceLimitJointData, cap, "dl_accumulated_impulse")
        dl_bias = schema_col_base(DistanceLimitJointData, cap, "dl_bias")
        dl_j0 = schema_col_base(DistanceLimitJointData, cap, "dl_j0")
        dl_j1 = schema_col_base(DistanceLimitJointData, cap, "dl_j1")
        dl_j2 = schema_col_base(DistanceLimitJointData, cap, "dl_j2")
        dl_j3 = schema_col_base(DistanceLimitJointData, cap, "dl_j3")
        dl_clamp = schema_col_base(DistanceLimitJointData, cap, "dl_clamp")

        # -- Body column base indices -------------------------------------
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

        # ===============================================================
        # Building-block prepare functions
        # Matches C# ConstraintPart.CalculateConstraintProperties + WarmStart
        # ===============================================================

        @wp.func
        def _prepare_point_constraint(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            rw0: wp.vec3,
            rw1: wp.vec3,
            inv_m0: float,
            inv_m1: float,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
        ):
            r0x = skew_symmetric(rw0)
            r1x = skew_symmetric(rw1)
            identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            k_inv = identity * (inv_m0 + inv_m1)
            k_inv = k_inv + r0x * inv_i0 * wp.transpose(r0x)
            k_inv = k_inv + r1x * inv_i1 * wp.transpose(r1x)
            ds_store_mat33(jdata, wp.static(j_point_eff_mass), ji, invert_mat33_safe(k_inv))

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

        @wp.func
        def _prepare_hinge_rotation(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            q0: wp.quat,
            q1: wp.quat,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
        ):
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

        # ===============================================================
        # Building-block solve functions
        # Matches C# ConstraintPart.SolveVelocityConstraint
        # ===============================================================

        @wp.func
        def _solve_point_constraint(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            v0: wp.vec3,
            w0: wp.vec3,
            v1: wp.vec3,
            w1: wp.vec3,
            inv_m0: float,
            inv_m1: float,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
            use_bias: int,
        ):
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

            return v0, w0, v1, w1

        @wp.func
        def _solve_hinge_rotation(
            jdata: wp.array(dtype=wp.float32),
            ji: int,
            w0: wp.vec3,
            w1: wp.vec3,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
        ):
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

            ds_store_float(
                jdata, wp.static(j_hinge_lambda_x), ji, ds_load_float(jdata, wp.static(j_hinge_lambda_x), ji) + lx
            )
            ds_store_float(
                jdata, wp.static(j_hinge_lambda_y), ji, ds_load_float(jdata, wp.static(j_hinge_lambda_y), ji) + ly
            )

            h_imp = b2xa1 * lx + c2xa1 * ly
            if not is_static_0:
                w0 = w0 - inv_i0 * h_imp
            if not is_static_1:
                w1 = w1 + inv_i1 * h_imp

            return w0, w1

        # ===============================================================
        # Per-type prepare functions
        # Matches C# joint PrepareForIteration methods
        # ===============================================================

        @wp.func
        def _prepare_ball_socket_joint(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            q0: wp.quat,
            q1: wp.quat,
            rw0: wp.vec3,
            rw1: wp.vec3,
            inv_m0: float,
            inv_m1: float,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
        ):
            _prepare_point_constraint(
                jdata, bdata, ji, b0, b1, rw0, rw1, inv_m0, inv_m1, inv_i0, inv_i1, is_static_0, is_static_1
            )

        @wp.func
        def _prepare_fixed_joint(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            q0: wp.quat,
            q1: wp.quat,
            rw0: wp.vec3,
            rw1: wp.vec3,
            inv_m0: float,
            inv_m1: float,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
        ):
            _prepare_point_constraint(
                jdata, bdata, ji, b0, b1, rw0, rw1, inv_m0, inv_m1, inv_i0, inv_i1, is_static_0, is_static_1
            )
            _prepare_hinge_rotation(jdata, bdata, ji, b0, b1, q0, q1, inv_i0, inv_i1, is_static_0, is_static_1)

        @wp.func
        def _prepare_revolute_joint(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            q0: wp.quat,
            q1: wp.quat,
            rw0: wp.vec3,
            rw1: wp.vec3,
            inv_m0: float,
            inv_m1: float,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
        ):
            _prepare_point_constraint(
                jdata, bdata, ji, b0, b1, rw0, rw1, inv_m0, inv_m1, inv_i0, inv_i1, is_static_0, is_static_1
            )
            _prepare_hinge_rotation(jdata, bdata, ji, b0, b1, q0, q1, inv_i0, inv_i1, is_static_0, is_static_1)

            # --- Angle constraint ---
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

            # --- Drive (revolute) ---
            dm = ds_load_int(jdata, wp.static(j_drive_mode), ji)
            if dm > 0:
                axis_d = ds_load_vec3(jdata, wp.static(j_angle_axis), ji)
                inv_eff_d = wp.dot(axis_d, (inv_i0 + inv_i1) * axis_d)

                d_damp_val = ds_load_float(jdata, wp.static(j_drive_damping), ji)
                compliance = 0.0
                if d_damp_val > 1.0e-6:
                    compliance = 1.0 / d_damp_val
                reg_inv = inv_eff_d + compliance
                if reg_inv > 0.0:
                    ds_store_float(jdata, wp.static(j_drive_eff_mass), ji, 1.0 / reg_inv)
                else:
                    ds_store_float(jdata, wp.static(j_drive_eff_mass), ji, 0.0)

                dl = ds_load_float(jdata, wp.static(j_drive_lambda), ji)
                d_imp = axis_d * dl
                if not is_static_0:
                    w0_d = ds_load_vec3(bdata, wp.static(b_angular_velocity), b0)
                    ds_store_vec3(bdata, wp.static(b_angular_velocity), b0, w0_d - inv_i0 * d_imp)
                if not is_static_1:
                    w1_d = ds_load_vec3(bdata, wp.static(b_angular_velocity), b1)
                    ds_store_vec3(bdata, wp.static(b_angular_velocity), b1, w1_d + inv_i1 * d_imp)

        @wp.func
        def _prepare_prismatic_joint(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            q0: wp.quat,
            q1: wp.quat,
            rw0: wp.vec3,
            rw1: wp.vec3,
            inv_m0: float,
            inv_m1: float,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
            inv_dt: float,
        ):
            _prepare_hinge_rotation(jdata, bdata, ji, b0, b1, q0, q1, inv_i0, inv_i1, is_static_0, is_static_1)

            # --- Slide perpendicular + axis ---
            a1 = wp.quat_rotate(q0, ds_load_vec3(jdata, wp.static(j_local_axis0), ji))
            ds_store_vec3(jdata, wp.static(j_slide_axis), ji, a1)

            perp0 = get_perpendiculars(a1)
            perp1 = wp.cross(a1, perp0)
            ds_store_vec3(jdata, wp.static(j_slide_perp0), ji, perp0)
            ds_store_vec3(jdata, wp.static(j_slide_perp1), ji, perp1)

            p0 = ds_load_vec3(bdata, wp.static(b_position), b0)
            p1 = ds_load_vec3(bdata, wp.static(b_position), b1)
            d_vec = (p1 + rw1) - (p0 + rw0)
            ds_store_float(jdata, wp.static(j_slide_current), ji, wp.dot(a1, d_vec))

            r0x = skew_symmetric(rw0)
            r1x = skew_symmetric(rw1)
            k_full = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) * (inv_m0 + inv_m1)
            k_full = k_full + r0x * inv_i0 * wp.transpose(r0x)
            k_full = k_full + r1x * inv_i1 * wp.transpose(r1x)

            kp00 = wp.dot(perp0, k_full * perp0)
            kp01 = wp.dot(perp0, k_full * perp1)
            kp10 = wp.dot(perp1, k_full * perp0)
            kp11 = wp.dot(perp1, k_full * perp1)
            inv_kp = invert_2x2(kp00, kp01, kp10, kp11)
            ds_store_float(jdata, wp.static(j_slide_perp_eff_00), ji, inv_kp[0])
            ds_store_float(jdata, wp.static(j_slide_perp_eff_01), ji, inv_kp[1])
            ds_store_float(jdata, wp.static(j_slide_perp_eff_10), ji, inv_kp[2])
            ds_store_float(jdata, wp.static(j_slide_perp_eff_11), ji, inv_kp[3])

            ka = wp.dot(a1, k_full * a1)
            if ka > 0.0:
                ds_store_float(jdata, wp.static(j_slide_eff_mass), ji, 1.0 / ka)
            else:
                ds_store_float(jdata, wp.static(j_slide_eff_mass), ji, 0.0)

            # Warm start perpendicular
            slx = ds_load_float(jdata, wp.static(j_slide_perp_lambda_x), ji)
            sly = ds_load_float(jdata, wp.static(j_slide_perp_lambda_y), ji)
            perp_imp = perp0 * slx + perp1 * sly
            if not is_static_0:
                v0_p = ds_load_vec3(bdata, wp.static(b_velocity), b0)
                w0_p = ds_load_vec3(bdata, wp.static(b_angular_velocity), b0)
                ds_store_vec3(bdata, wp.static(b_velocity), b0, v0_p - inv_m0 * perp_imp)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b0, w0_p - inv_i0 * wp.cross(rw0, perp_imp))
            if not is_static_1:
                v1_p = ds_load_vec3(bdata, wp.static(b_velocity), b1)
                w1_p = ds_load_vec3(bdata, wp.static(b_angular_velocity), b1)
                ds_store_vec3(bdata, wp.static(b_velocity), b1, v1_p + inv_m1 * perp_imp)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b1, w1_p + inv_i1 * wp.cross(rw1, perp_imp))

            # --- Drive (prismatic) — Jolt soft constraint (Erin Catto GDC 2011) ---
            dm = ds_load_int(jdata, wp.static(j_drive_mode), ji)
            if dm > 0:
                axis_d = ds_load_vec3(jdata, wp.static(j_slide_axis), ji)
                inv_eff_d = inv_m0 + inv_m1

                d_stiff = ds_load_float(jdata, wp.static(j_drive_stiffness), ji)
                d_damp = ds_load_float(jdata, wp.static(j_drive_damping), ji)
                dt = 1.0 / wp.max(inv_dt, 1.0e-6)

                if d_stiff > 0.0 or d_damp > 0.0:
                    # softness = 1/(dt*(c + dt*k)),  eff_mass = 1/(inv_eff + softness)
                    softness = 1.0 / (dt * (d_damp + dt * d_stiff))
                    ds_store_float(jdata, wp.static(j_drive_eff_mass), ji, 1.0 / (inv_eff_d + softness))
                else:
                    ds_store_float(jdata, wp.static(j_drive_eff_mass), ji, 0.0)

                dl = ds_load_float(jdata, wp.static(j_drive_lambda), ji)
                d_imp_lin = axis_d * dl
                if not is_static_0:
                    v0_d = ds_load_vec3(bdata, wp.static(b_velocity), b0)
                    ds_store_vec3(bdata, wp.static(b_velocity), b0, v0_d - inv_m0 * d_imp_lin)
                if not is_static_1:
                    v1_d = ds_load_vec3(bdata, wp.static(b_velocity), b1)
                    ds_store_vec3(bdata, wp.static(b_velocity), b1, v1_d + inv_m1 * d_imp_lin)

        # ===============================================================
        # Per-type solve functions
        # Matches C# joint Iterate methods
        # ===============================================================

        @wp.func
        def _solve_ball_socket_joint(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            v0: wp.vec3,
            w0: wp.vec3,
            v1: wp.vec3,
            w1: wp.vec3,
            inv_m0: float,
            inv_m1: float,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
            use_bias: int,
        ):
            v0, w0, v1, w1 = _solve_point_constraint(
                jdata,
                bdata,
                ji,
                b0,
                b1,
                v0,
                w0,
                v1,
                w1,
                inv_m0,
                inv_m1,
                inv_i0,
                inv_i1,
                is_static_0,
                is_static_1,
                use_bias,
            )
            return v0, w0, v1, w1

        @wp.func
        def _solve_fixed_joint(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            v0: wp.vec3,
            w0: wp.vec3,
            v1: wp.vec3,
            w1: wp.vec3,
            inv_m0: float,
            inv_m1: float,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
            use_bias: int,
        ):
            w0, w1 = _solve_hinge_rotation(jdata, ji, w0, w1, inv_i0, inv_i1, is_static_0, is_static_1)
            v0, w0, v1, w1 = _solve_point_constraint(
                jdata,
                bdata,
                ji,
                b0,
                b1,
                v0,
                w0,
                v1,
                w1,
                inv_m0,
                inv_m1,
                inv_i0,
                inv_i1,
                is_static_0,
                is_static_1,
                use_bias,
            )
            return v0, w0, v1, w1

        @wp.func
        def _solve_revolute_joint(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            v0: wp.vec3,
            w0: wp.vec3,
            v1: wp.vec3,
            w1: wp.vec3,
            inv_m0: float,
            inv_m1: float,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
            use_bias: int,
        ):
            # 1. Hinge rotation (C# rotationConstraintPart.SolveVelocityConstraint)
            w0, w1 = _solve_hinge_rotation(jdata, ji, w0, w1, inv_i0, inv_i1, is_static_0, is_static_1)

            # 2. Angle constraint with limits (C# motorConstraintPart.SolveVelocityConstraint)
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

            # 3. Drive (revolute — angular)
            d_eff = ds_load_float(jdata, wp.static(j_drive_eff_mass), ji)
            dm = ds_load_int(jdata, wp.static(j_drive_mode), ji)
            if dm > 0 and d_eff > 0.0:
                d_target = ds_load_float(jdata, wp.static(j_drive_target), ji)
                d_stiff = ds_load_float(jdata, wp.static(j_drive_stiffness), ji)
                d_damp = ds_load_float(jdata, wp.static(j_drive_damping), ji)
                d_max = ds_load_float(jdata, wp.static(j_drive_max_force), ji)

                axis_d = ds_load_vec3(jdata, wp.static(j_angle_axis), ji)
                jv_d = wp.dot(axis_d, w0 - w1)
                current_val = ds_load_float(jdata, wp.static(j_angle_current), ji)

                ratio = 0.0
                if d_damp > 1.0e-6:
                    ratio = d_stiff / d_damp

                dl = 0.0
                if dm == DRIVE_POSITION:
                    dl = d_eff * (jv_d - ratio * (current_val - d_target))
                elif dm == DRIVE_VELOCITY:
                    dl = d_eff * (jv_d + d_target)

                old_dl = ds_load_float(jdata, wp.static(j_drive_lambda), ji)
                new_dl = wp.clamp(old_dl + dl, -d_max, d_max)
                dl = new_dl - old_dl
                ds_store_float(jdata, wp.static(j_drive_lambda), ji, new_dl)

                d_imp = axis_d * dl
                if not is_static_0:
                    w0 = w0 - inv_i0 * d_imp
                if not is_static_1:
                    w1 = w1 + inv_i1 * d_imp

            # 4. Point-on-point (C# pointConstraintPart.SolveVelocityConstraint)
            v0, w0, v1, w1 = _solve_point_constraint(
                jdata,
                bdata,
                ji,
                b0,
                b1,
                v0,
                w0,
                v1,
                w1,
                inv_m0,
                inv_m1,
                inv_i0,
                inv_i1,
                is_static_0,
                is_static_1,
                use_bias,
            )

            return v0, w0, v1, w1

        @wp.func
        def _solve_prismatic_joint(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            v0: wp.vec3,
            w0: wp.vec3,
            v1: wp.vec3,
            w1: wp.vec3,
            inv_m0: float,
            inv_m1: float,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
            use_bias: int,
            inv_dt: float,
        ):
            # 1. Hinge rotation
            w0, w1 = _solve_hinge_rotation(jdata, ji, w0, w1, inv_i0, inv_i1, is_static_0, is_static_1)

            # 2. Slide perpendicular (C# DualAxisConstraintPart)
            perp0 = ds_load_vec3(jdata, wp.static(j_slide_perp0), ji)
            perp1 = ds_load_vec3(jdata, wp.static(j_slide_perp1), ji)

            rw0_s = ds_load_vec3(jdata, wp.static(j_rw0), ji)
            rw1_s = ds_load_vec3(jdata, wp.static(j_rw1), ji)
            # C# DualAxisConstraintPart uses delta_lin = body1.Vel - body2.Vel
            # which maps to (v0 - v1) in our naming convention.
            dv = (v0 + wp.cross(w0, rw0_s)) - (v1 + wp.cross(w1, rw1_s))

            jv_px = wp.dot(perp0, dv)
            jv_py = wp.dot(perp1, dv)

            perp_bias_x = 0.0
            perp_bias_y = 0.0
            if use_bias != 0:
                p0 = ds_load_vec3(bdata, wp.static(b_position), b0)
                p1 = ds_load_vec3(bdata, wp.static(b_position), b1)
                d_vec = (p1 + rw1_s) - (p0 + rw0_s)
                perp_bias_x = -CONSTRAINT_BAUMGARTE * wp.dot(perp0, d_vec)
                perp_bias_y = -CONSTRAINT_BAUMGARTE * wp.dot(perp1, d_vec)

            ep00 = ds_load_float(jdata, wp.static(j_slide_perp_eff_00), ji)
            ep01 = ds_load_float(jdata, wp.static(j_slide_perp_eff_01), ji)
            ep10 = ds_load_float(jdata, wp.static(j_slide_perp_eff_10), ji)
            ep11 = ds_load_float(jdata, wp.static(j_slide_perp_eff_11), ji)

            rhs_x = jv_px + perp_bias_x
            rhs_y = jv_py + perp_bias_y
            lx = ep00 * rhs_x + ep01 * rhs_y
            ly = ep10 * rhs_x + ep11 * rhs_y

            ds_store_float(
                jdata,
                wp.static(j_slide_perp_lambda_x),
                ji,
                ds_load_float(jdata, wp.static(j_slide_perp_lambda_x), ji) + lx,
            )
            ds_store_float(
                jdata,
                wp.static(j_slide_perp_lambda_y),
                ji,
                ds_load_float(jdata, wp.static(j_slide_perp_lambda_y), ji) + ly,
            )

            perp_imp = perp0 * lx + perp1 * ly
            if not is_static_0:
                v0 = v0 - inv_m0 * perp_imp
                w0 = w0 - inv_i0 * wp.cross(rw0_s, perp_imp)
            if not is_static_1:
                v1 = v1 + inv_m1 * perp_imp
                w1 = w1 + inv_i1 * wp.cross(rw1_s, perp_imp)

            # 3. Slide axis limits (C# AxisConstraintBlock)
            s_eff = ds_load_float(jdata, wp.static(j_slide_eff_mass), ji)
            s_min = ds_load_float(jdata, wp.static(j_slide_min), ji)
            s_max = ds_load_float(jdata, wp.static(j_slide_max), ji)
            has_slide_limits = s_min > -1.0e6 and s_max < 1.0e6

            if has_slide_limits and s_eff > 0.0:
                s_cur = ds_load_float(jdata, wp.static(j_slide_current), ji)

                # Only activate when at or beyond limits
                if s_cur <= s_min or s_cur >= s_max:
                    s_axis = ds_load_vec3(jdata, wp.static(j_slide_axis), ji)
                    rw0_sl = ds_load_vec3(jdata, wp.static(j_rw0), ji)
                    rw1_sl = ds_load_vec3(jdata, wp.static(j_rw1), ji)
                    dv_s = (v1 + wp.cross(w1, rw1_sl)) - (v0 + wp.cross(w0, rw0_sl))
                    jv_s = wp.dot(s_axis, dv_s)

                    s_bias = 0.0
                    if use_bias != 0:
                        if s_cur <= s_min:
                            s_bias = CONSTRAINT_BAUMGARTE * (s_cur - s_min)
                        elif s_cur >= s_max:
                            s_bias = CONSTRAINT_BAUMGARTE * (s_cur - s_max)

                    sl = -s_eff * (jv_s + s_bias)
                    old_sl = ds_load_float(jdata, wp.static(j_slide_lambda), ji)
                    new_sl = old_sl + sl

                    if s_cur <= s_min:
                        new_sl = wp.max(new_sl, 0.0)
                    else:
                        new_sl = wp.min(new_sl, 0.0)

                    sl = new_sl - old_sl
                    ds_store_float(jdata, wp.static(j_slide_lambda), ji, new_sl)

                s_imp = s_axis * sl
                if not is_static_0:
                    v0 = v0 - inv_m0 * s_imp
                    w0 = w0 - inv_i0 * wp.cross(rw0_sl, s_imp)
                if not is_static_1:
                    v1 = v1 + inv_m1 * s_imp
                    w1 = w1 + inv_i1 * wp.cross(rw1_sl, s_imp)

            # 4. Drive (prismatic — Jolt soft constraint)
            d_eff = ds_load_float(jdata, wp.static(j_drive_eff_mass), ji)
            dm = ds_load_int(jdata, wp.static(j_drive_mode), ji)
            if dm > 0 and d_eff > 0.0:
                d_target = ds_load_float(jdata, wp.static(j_drive_target), ji)
                d_stiff = ds_load_float(jdata, wp.static(j_drive_stiffness), ji)
                d_damp = ds_load_float(jdata, wp.static(j_drive_damping), ji)
                d_max = ds_load_float(jdata, wp.static(j_drive_max_force), ji)

                axis_d = ds_load_vec3(jdata, wp.static(j_slide_axis), ji)
                rw0_d = ds_load_vec3(jdata, wp.static(j_rw0), ji)
                rw1_d = ds_load_vec3(jdata, wp.static(j_rw1), ji)
                dv_d = (v1 + wp.cross(w1, rw1_d)) - (v0 + wp.cross(w0, rw0_d))
                jv_d = wp.dot(axis_d, dv_d)
                current_val = ds_load_float(jdata, wp.static(j_slide_current), ji)

                old_dl = ds_load_float(jdata, wp.static(j_drive_lambda), ji)

                # Jolt soft constraint: lambda = -eff_mass * (Jv + GetBias)
                # softness = 1/(dt*(c + dt*k)), bias_factor = dt*k*softness
                # GetBias(total_lambda) = softness * total_lambda + bias_factor * C
                dl = 0.0
                if dm == DRIVE_POSITION:
                    dt = 1.0 / wp.max(inv_dt, 1.0e-6)
                    error = current_val - d_target
                    if d_stiff > 0.0 or d_damp > 0.0:
                        softness = 1.0 / (dt * (d_damp + dt * d_stiff))
                        bias_factor = dt * d_stiff * softness
                        bias = softness * old_dl + bias_factor * error
                        dl = -d_eff * (jv_d + bias)
                elif dm == DRIVE_VELOCITY:
                    dl = -d_eff * (jv_d - d_target)

                new_dl = wp.clamp(old_dl + dl, -d_max, d_max)
                dl = new_dl - old_dl
                ds_store_float(jdata, wp.static(j_drive_lambda), ji, new_dl)

                d_imp_lin = axis_d * dl
                if not is_static_0:
                    v0 = v0 - inv_m0 * d_imp_lin
                if not is_static_1:
                    v1 = v1 + inv_m1 * d_imp_lin

            return v0, w0, v1, w1

        # ===============================================================
        # Distance limit prepare / solve
        # ===============================================================

        @wp.func
        def _prepare_distance_limit(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            q0: wp.quat,
            q1: wp.quat,
            rw0: wp.vec3,
            rw1: wp.vec3,
            inv_m0: float,
            inv_m1: float,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
            inv_dt: float,
        ):
            # World positions
            pos0 = ds_load_vec3(bdata, wp.static(b_position), b0)
            pos1 = ds_load_vec3(bdata, wp.static(b_position), b1)
            p0 = pos0 + rw0
            p1 = pos1 + rw1

            # Distance and direction
            dp = p1 - p0
            dist = wp.length(dp)
            rest = ds_load_float(jdata, wp.static(dl_rest_distance), ji)
            error = dist - rest

            lim_min = ds_load_float(jdata, wp.static(dl_limit_min), ji)
            lim_max = ds_load_float(jdata, wp.static(dl_limit_max), ji)

            # Clamp check: 1=max violated (stretched), 2=min violated (compressed)
            clamp_mode = int(0)
            error_from_limit = float(0.0)
            if error > lim_max:
                clamp_mode = int(1)
                error_from_limit = float(error - lim_max)
            elif error < lim_min:
                clamp_mode = int(2)
                error_from_limit = float(lim_min - error)  # positive = compression

            ds_store_int(jdata, wp.static(dl_clamp), ji, clamp_mode)

            if clamp_mode == 0:
                # Inactive - reset accumulated impulse
                ds_store_float(jdata, wp.static(dl_accumulated_impulse), ji, 0.0)
                return

            # Normalize direction (avoid division by zero)
            n = wp.vec3(0.0, 1.0, 0.0)
            if dist > 1.0e-6:
                n = dp / dist

            # Jacobians: J = [-n, -rw0×n, n, rw1×n]
            j0_val = -n
            j1_val = -wp.cross(rw0, n)
            j2_val = n
            j3_val = wp.cross(rw1, n)
            ds_store_vec3(jdata, wp.static(dl_j0), ji, j0_val)
            ds_store_vec3(jdata, wp.static(dl_j1), ji, j1_val)
            ds_store_vec3(jdata, wp.static(dl_j2), ji, j2_val)
            ds_store_vec3(jdata, wp.static(dl_j3), ji, j3_val)

            # Effective mass: 1 / (J M^-1 J^T + softness)
            ang0 = inv_i0 * j1_val
            ang1 = inv_i1 * j3_val
            inv_eff = inv_m0 + inv_m1 + wp.dot(j1_val, ang0) + wp.dot(j3_val, ang1)

            soft = ds_load_float(jdata, wp.static(dl_softness), ji)
            # C# multiplies softness by inv_dt in both effective mass and
            # the solve softness scalar (impulse-based formulation).
            soft_scaled = soft * inv_dt
            inv_eff = inv_eff + soft_scaled

            eff = 0.0
            if inv_eff > 0.0:
                eff = 1.0 / inv_eff
            ds_store_float(jdata, wp.static(dl_eff_mass), ji, eff)
            # Store soft_scaled in bias_factor (no longer needed after bias computation below)
            # so that _solve_distance_limit can read it without inv_dt.

            # Bias: error * bias_factor * inv_dt
            bf = ds_load_float(jdata, wp.static(dl_bias_factor), ji)
            if clamp_mode == 2:
                # Compression: bias drives bodies apart
                bias_val = -error_from_limit * bf * inv_dt
            else:
                # Tension: bias drives bodies together
                bias_val = error_from_limit * bf * inv_dt
            ds_store_float(jdata, wp.static(dl_bias), ji, bias_val)
            # Repurpose bias_factor slot to store soft_scaled for the solve phase
            ds_store_float(jdata, wp.static(dl_bias_factor), ji, soft_scaled)

            # Warm start
            acc = ds_load_float(jdata, wp.static(dl_accumulated_impulse), ji)
            if not is_static_0:
                v0 = ds_load_vec3(bdata, wp.static(b_velocity), b0)
                w0 = ds_load_vec3(bdata, wp.static(b_angular_velocity), b0)
                ds_store_vec3(bdata, wp.static(b_velocity), b0, v0 + inv_m0 * acc * j0_val)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b0, w0 + inv_i0 * (acc * j1_val))
            if not is_static_1:
                v1 = ds_load_vec3(bdata, wp.static(b_velocity), b1)
                w1 = ds_load_vec3(bdata, wp.static(b_angular_velocity), b1)
                ds_store_vec3(bdata, wp.static(b_velocity), b1, v1 + inv_m1 * acc * j2_val)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b1, w1 + inv_i1 * (acc * j3_val))

        @wp.func
        def _solve_distance_limit(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            ji: int,
            b0: int,
            b1: int,
            v0: wp.vec3,
            w0: wp.vec3,
            v1: wp.vec3,
            w1: wp.vec3,
            inv_m0: float,
            inv_m1: float,
            inv_i0: wp.mat33,
            inv_i1: wp.mat33,
            is_static_0: bool,
            is_static_1: bool,
            use_bias: int,
        ):
            clamp_mode = ds_load_int(jdata, wp.static(dl_clamp), ji)
            if clamp_mode == 0:
                return v0, w0, v1, w1

            j0_val = ds_load_vec3(jdata, wp.static(dl_j0), ji)
            j1_val = ds_load_vec3(jdata, wp.static(dl_j1), ji)
            j2_val = ds_load_vec3(jdata, wp.static(dl_j2), ji)
            j3_val = ds_load_vec3(jdata, wp.static(dl_j3), ji)

            # Relative velocity along constraint
            jv = wp.dot(j0_val, v0) + wp.dot(j1_val, w0) + wp.dot(j2_val, v1) + wp.dot(j3_val, w1)

            eff = ds_load_float(jdata, wp.static(dl_eff_mass), ji)
            acc = ds_load_float(jdata, wp.static(dl_accumulated_impulse), ji)

            # Softness (spring damping): soft_scaled * accumulated_impulse
            # soft_scaled = softness * inv_dt, stored in bias_factor slot by prepare
            soft_scaled = ds_load_float(jdata, wp.static(dl_bias_factor), ji)
            softness_scalar = acc * soft_scaled

            # Bias
            bias_val = 0.0
            if use_bias != 0:
                bias_val = ds_load_float(jdata, wp.static(dl_bias), ji)

            # Compute impulse
            delta = -eff * (jv + bias_val + softness_scalar)

            # Accumulate with directional clamping
            old_acc = acc
            new_acc = old_acc + delta
            if clamp_mode == 1:
                # Max violated (stretched) — only pull inward (negative impulse)
                new_acc = wp.min(new_acc, 0.0)
            else:
                # Min violated (compressed) — only push outward (positive impulse)
                new_acc = wp.max(new_acc, 0.0)
            ds_store_float(jdata, wp.static(dl_accumulated_impulse), ji, new_acc)
            applied = new_acc - old_acc

            # Apply to velocities
            if not is_static_0:
                v0 = v0 + inv_m0 * applied * j0_val
                w0 = w0 + inv_i0 * (applied * j1_val)
            if not is_static_1:
                v1 = v1 + inv_m1 * applied * j2_val
                w1 = w1 + inv_i1 * (applied * j3_val)

            return v0, w0, v1, w1

        # ===============================================================
        # Prepare kernel (thin dispatcher)
        # ===============================================================

        @wp.kernel
        def _prepare(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            partition_data: wp.array(dtype=wp.int32),
            partition_end_arr: wp.array(dtype=wp.int32),
            partition_slot: int,
            contact_count_arr: wp.array(dtype=wp.int32),
            joint_count: wp.array(dtype=wp.int32),
            inv_dt: float,
        ):
            tid = wp.tid()
            p_start = int(0)
            if partition_slot > 0:
                p_start = partition_end_arr[partition_slot - 1]
            p_end = partition_end_arr[partition_slot]
            if tid >= p_end - p_start:
                return

            contact_count = contact_count_arr[0]
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

            # World-space anchor offsets (common to all types)
            rw0 = wp.quat_rotate(q0, ds_load_vec3(jdata, wp.static(j_local_anchor0), ji))
            rw1 = wp.quat_rotate(q1, ds_load_vec3(jdata, wp.static(j_local_anchor1), ji))
            ds_store_vec3(jdata, wp.static(j_rw0), ji, rw0)
            ds_store_vec3(jdata, wp.static(j_rw1), ji, rw1)

            if jtype == JOINT_BALL_SOCKET:
                _prepare_ball_socket_joint(
                    jdata, bdata, ji, b0, b1, q0, q1, rw0, rw1, inv_m0, inv_m1, inv_i0, inv_i1, is_static_0, is_static_1
                )
            elif jtype == JOINT_REVOLUTE:
                _prepare_revolute_joint(
                    jdata, bdata, ji, b0, b1, q0, q1, rw0, rw1, inv_m0, inv_m1, inv_i0, inv_i1, is_static_0, is_static_1
                )
            elif jtype == JOINT_FIXED:
                _prepare_fixed_joint(
                    jdata, bdata, ji, b0, b1, q0, q1, rw0, rw1, inv_m0, inv_m1, inv_i0, inv_i1, is_static_0, is_static_1
                )
            elif jtype == JOINT_PRISMATIC:
                _prepare_prismatic_joint(
                    jdata,
                    bdata,
                    ji,
                    b0,
                    b1,
                    q0,
                    q1,
                    rw0,
                    rw1,
                    inv_m0,
                    inv_m1,
                    inv_i0,
                    inv_i1,
                    is_static_0,
                    is_static_1,
                    inv_dt,
                )
            elif jtype == JOINT_DISTANCE_LIMIT:
                _prepare_distance_limit(
                    jdata,
                    bdata,
                    ji,
                    b0,
                    b1,
                    q0,
                    q1,
                    rw0,
                    rw1,
                    inv_m0,
                    inv_m1,
                    inv_i0,
                    inv_i1,
                    is_static_0,
                    is_static_1,
                    inv_dt,
                )

        # ===============================================================
        # Solve kernel (thin dispatcher)
        # ===============================================================

        @wp.kernel
        def _solve(
            jdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            partition_data: wp.array(dtype=wp.int32),
            partition_end_arr: wp.array(dtype=wp.int32),
            partition_slot: int,
            contact_count_arr: wp.array(dtype=wp.int32),
            joint_count: wp.array(dtype=wp.int32),
            use_bias: int,
            inv_dt: float,
        ):
            tid = wp.tid()
            p_start = int(0)
            if partition_slot > 0:
                p_start = partition_end_arr[partition_slot - 1]
            p_end = partition_end_arr[partition_slot]
            if tid >= p_end - p_start:
                return

            contact_count = contact_count_arr[0]
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

            if jtype == JOINT_BALL_SOCKET:
                v0, w0, v1, w1 = _solve_ball_socket_joint(
                    jdata,
                    bdata,
                    ji,
                    b0,
                    b1,
                    v0,
                    w0,
                    v1,
                    w1,
                    inv_m0,
                    inv_m1,
                    inv_i0,
                    inv_i1,
                    is_static_0,
                    is_static_1,
                    use_bias,
                )
            elif jtype == JOINT_REVOLUTE:
                v0, w0, v1, w1 = _solve_revolute_joint(
                    jdata,
                    bdata,
                    ji,
                    b0,
                    b1,
                    v0,
                    w0,
                    v1,
                    w1,
                    inv_m0,
                    inv_m1,
                    inv_i0,
                    inv_i1,
                    is_static_0,
                    is_static_1,
                    use_bias,
                )
            elif jtype == JOINT_FIXED:
                v0, w0, v1, w1 = _solve_fixed_joint(
                    jdata,
                    bdata,
                    ji,
                    b0,
                    b1,
                    v0,
                    w0,
                    v1,
                    w1,
                    inv_m0,
                    inv_m1,
                    inv_i0,
                    inv_i1,
                    is_static_0,
                    is_static_1,
                    use_bias,
                )
            elif jtype == JOINT_PRISMATIC:
                v0, w0, v1, w1 = _solve_prismatic_joint(
                    jdata,
                    bdata,
                    ji,
                    b0,
                    b1,
                    v0,
                    w0,
                    v1,
                    w1,
                    inv_m0,
                    inv_m1,
                    inv_i0,
                    inv_i1,
                    is_static_0,
                    is_static_1,
                    use_bias,
                    inv_dt,
                )
            elif jtype == JOINT_DISTANCE_LIMIT:
                v0, w0, v1, w1 = _solve_distance_limit(
                    jdata,
                    bdata,
                    ji,
                    b0,
                    b1,
                    v0,
                    w0,
                    v1,
                    w1,
                    inv_m0,
                    inv_m1,
                    inv_i0,
                    inv_i1,
                    is_static_0,
                    is_static_1,
                    use_bias,
                )

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
