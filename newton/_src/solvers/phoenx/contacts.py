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

"""PhoenX PGS contact constraint kernels: import, prepare, solve."""

from __future__ import annotations

import warp as wp

from .constraints import (
    col_base,
    ds_load_float,
    ds_load_int,
    ds_load_mat33,
    ds_load_quat,
    ds_load_vec3,
    ds_store_float,
    ds_store_vec3,
)
from .schemas import BODY_FLAG_STATIC

# ---------------------------------------------------------------------------
# Constants (matching C# PhoenX)
# ---------------------------------------------------------------------------

BAUMGARTE_FACTOR = wp.constant(0.03)
MAX_BIAS = wp.constant(100.0)
WARM_START_SCALE = wp.constant(0.90)


# ---------------------------------------------------------------------------
# Contact import
# ---------------------------------------------------------------------------


@wp.kernel
def import_contacts_kernel(
    n_shape0: wp.array(dtype=wp.int32),
    n_shape1: wp.array(dtype=wp.int32),
    n_normal: wp.array(dtype=wp.vec3),
    n_offset0: wp.array(dtype=wp.vec3),
    n_offset1: wp.array(dtype=wp.vec3),
    n_margin0: wp.array(dtype=wp.float32),
    n_margin1: wp.array(dtype=wp.float32),
    contact_count: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    default_friction: float,
    out_shape0: wp.array(dtype=wp.int32),
    out_shape1: wp.array(dtype=wp.int32),
    out_body0: wp.array(dtype=wp.int32),
    out_body1: wp.array(dtype=wp.int32),
    out_normal: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_margin0: wp.array(dtype=wp.float32),
    out_margin1: wp.array(dtype=wp.float32),
    out_friction: wp.array(dtype=wp.float32),
):
    """Copy Newton rigid contacts into PhoenX contact DataStore columns."""
    tid = wp.tid()
    if tid >= contact_count[0]:
        return
    s0 = n_shape0[tid]
    s1 = n_shape1[tid]
    out_shape0[tid] = s0
    out_shape1[tid] = s1
    out_body0[tid] = shape_body[s0]
    out_body1[tid] = shape_body[s1]
    out_normal[tid] = n_normal[tid]
    out_offset0[tid] = n_offset0[tid]
    out_offset1[tid] = n_offset1[tid]
    out_margin0[tid] = n_margin0[tid]
    out_margin1[tid] = n_margin1[tid]
    out_friction[tid] = default_friction


# ---------------------------------------------------------------------------
# Solver device functions
# ---------------------------------------------------------------------------


@wp.func
def compute_tangent_frame(n: wp.vec3) -> wp.vec3:
    """Return a unit vector t1 orthogonal to *n*.  t2 = cross(t1, n)."""
    if wp.abs(n[0]) < 0.9:
        t = wp.cross(wp.vec3(1.0, 0.0, 0.0), n)
    else:
        t = wp.cross(wp.vec3(0.0, 1.0, 0.0), n)
    return wp.normalize(t)


@wp.func
def compute_effective_mass(
    inv_mass_a: float,
    inv_mass_b: float,
    inv_inertia_a: wp.mat33,
    inv_inertia_b: wp.mat33,
    rw_a: wp.vec3,
    rw_b: wp.vec3,
    direction: wp.vec3,
) -> float:
    """Diagonal effective mass: ``1 / (J M^-1 J^T)`` for one contact direction."""
    cross_a = wp.cross(rw_a, direction)
    cross_b = wp.cross(rw_b, direction)
    ang_a = inv_inertia_a * cross_a
    ang_b = inv_inertia_b * cross_b
    denom = inv_mass_a + inv_mass_b + wp.dot(cross_a, ang_a) + wp.dot(cross_b, ang_b)
    if denom > 0.0:
        return 1.0 / denom
    return 0.0


@wp.func
def compute_effective_mass_split(
    inv_mass_a: float,
    inv_mass_b: float,
    inv_inertia_a: wp.mat33,
    inv_inertia_b: wp.mat33,
    rw_a: wp.vec3,
    rw_b: wp.vec3,
    direction: wp.vec3,
    split_a: float,
    split_b: float,
) -> float:
    """Split-mass effective mass (Tonge et al. 2012).

    Each body's inverse-mass and inverse-inertia contributions are
    multiplied by the number of contacts acting on that body
    (``split_a``, ``split_b``).  This makes each contact "see" only
    its fraction of the body's mass, dramatically improving PGS
    convergence for stacked configurations while preserving momentum
    conservation (impulses are applied with the full, unsplit mass).
    """
    cross_a = wp.cross(rw_a, direction)
    cross_b = wp.cross(rw_b, direction)
    ang_a = inv_inertia_a * cross_a
    ang_b = inv_inertia_b * cross_b
    denom = (
        split_a * inv_mass_a
        + split_b * inv_mass_b
        + split_a * wp.dot(cross_a, ang_a)
        + split_b * wp.dot(cross_b, ang_b)
    )
    if denom > 0.0:
        return 1.0 / denom
    return 0.0


@wp.func
def compute_contact_bias(penetration: float, inv_dt: float) -> float:
    """Baumgarte stabilisation bias (matches C# ``ContactHelperGpu``)."""
    if penetration > 0.0:
        return wp.min(MAX_BIAS, BAUMGARTE_FACTOR * inv_dt * penetration)
    return 0.0


@wp.func
def apply_body_impulse(
    vel: wp.vec3,
    ang_vel: wp.vec3,
    inv_mass: float,
    inv_inertia: wp.mat33,
    rw: wp.vec3,
    impulse_world: wp.vec3,
    sign: float,
) -> wp.vec2i:
    """Apply *sign* * *impulse_world* to body velocity/angular velocity.

    Returns a dummy ``vec2i`` -- side effects are via the velocity arrays
    passed to the calling kernel; this function is inlined.
    """
    # Warp doesn't allow returning multiple values from a @wp.func in a way
    # that also writes to arrays, so we return the updated velocities via a
    # helper struct pattern below in the kernels instead.
    return wp.vec2i(0, 0)


# ---------------------------------------------------------------------------
# Build elements for bundle-based GraphColoring
# ---------------------------------------------------------------------------


@wp.kernel
def build_bundle_elements_kernel(
    bundle_starts: wp.array(dtype=wp.int32),
    sort_perm: wp.array(dtype=wp.int32),
    body0: wp.array(dtype=wp.int32),
    body1: wp.array(dtype=wp.int32),
    elements: wp.array2d(dtype=wp.int32),
    bundle_count: wp.array(dtype=wp.int32),
):
    """Build graph-coloring elements from contact bundles (one element per bundle)."""
    tid = wp.tid()
    if tid >= bundle_count[0]:
        return
    sorted_idx = bundle_starts[tid]
    ci = sort_perm[sorted_idx]
    elements[tid, 0] = body0[ci]
    elements[tid, 1] = body1[ci]
    for j in range(2, 8):
        elements[tid, j] = -1


# ---------------------------------------------------------------------------
# Mass-splitting: per-body contact count
# ---------------------------------------------------------------------------


@wp.kernel
def clear_contact_count_kernel(
    contact_count_per_body: wp.array(dtype=wp.int32),
    body_count: wp.array(dtype=wp.int32),
):
    """Zero the per-body contact counter."""
    tid = wp.tid()
    if tid >= body_count[0]:
        return
    contact_count_per_body[tid] = 0


@wp.kernel
def count_contacts_per_body_kernel(
    c_body0: wp.array(dtype=wp.int32),
    c_body1: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    contact_count_per_body: wp.array(dtype=wp.int32),
):
    """Atomically count how many contacts reference each body."""
    tid = wp.tid()
    if tid >= contact_count[0]:
        return
    wp.atomic_add(contact_count_per_body, c_body0[tid], 1)
    wp.atomic_add(contact_count_per_body, c_body1[tid], 1)


@wp.kernel
def count_partners_per_body_kernel(
    c_body0: wp.array(dtype=wp.int32),
    c_body1: wp.array(dtype=wp.int32),
    bundle_starts: wp.array(dtype=wp.int32),
    bundle_count: wp.array(dtype=wp.int32),
    sort_perm: wp.array(dtype=wp.int32),
    partner_count_per_body: wp.array(dtype=wp.int32),
):
    """Count bundles per body for mass splitting.

    Each bundle is an independent element in graph coloring and may
    share a body with other bundles in the same partition.  The bundle
    count per body is the correct Tonge mass-splitting factor: it
    equals the number of contacts per body divided by the bundle size,
    which is the effective parallelism the solver faces.
    """
    tid = wp.tid()
    if tid >= bundle_count[0]:
        return
    ci = sort_perm[bundle_starts[tid]]
    b0 = c_body0[ci]
    b1 = c_body1[ci]
    wp.atomic_add(partner_count_per_body, b0, 1)
    if b0 != b1:
        wp.atomic_add(partner_count_per_body, b1, 1)


# ---------------------------------------------------------------------------
# Prepare contacts kernel
# ---------------------------------------------------------------------------


@wp.kernel
def prepare_contacts_kernel(
    partition_data: wp.array(dtype=wp.int32),
    partition_end_arr: wp.array(dtype=wp.int32),
    partition_slot: int,
    c_normal: wp.array(dtype=wp.vec3),
    c_offset0: wp.array(dtype=wp.vec3),
    c_offset1: wp.array(dtype=wp.vec3),
    c_body0: wp.array(dtype=wp.int32),
    c_body1: wp.array(dtype=wp.int32),
    c_accumulated_n: wp.array(dtype=wp.float32),
    c_accumulated_t1: wp.array(dtype=wp.float32),
    c_accumulated_t2: wp.array(dtype=wp.float32),
    c_tangent1: wp.array(dtype=wp.vec3),
    c_rel0: wp.array(dtype=wp.vec3),
    c_rel1: wp.array(dtype=wp.vec3),
    c_eff_n: wp.array(dtype=wp.float32),
    c_eff_t1: wp.array(dtype=wp.float32),
    c_eff_t2: wp.array(dtype=wp.float32),
    c_bias: wp.array(dtype=wp.float32),
    c_margin0: wp.array(dtype=wp.float32),
    c_margin1: wp.array(dtype=wp.float32),
    b_position: wp.array(dtype=wp.vec3),
    b_orientation: wp.array(dtype=wp.quat),
    b_velocity: wp.array(dtype=wp.vec3),
    b_angular_velocity: wp.array(dtype=wp.vec3),
    b_inverse_mass: wp.array(dtype=wp.float32),
    b_inverse_inertia_world: wp.array(dtype=wp.mat33),
    b_flags: wp.array(dtype=wp.int32),
    contact_count_per_body: wp.array(dtype=wp.int32),
    inv_dt: float,
):
    """Compute per-contact solver data and apply warm-start impulses.

    Effective masses use **split** inverse masses (Tonge et al. 2012)
    for faster PGS convergence.  Warm-start impulses use the **full**
    inverse masses so that momentum is conserved.
    """
    tid = wp.tid()
    p_start = int(0)
    if partition_slot > 0:
        p_start = partition_end_arr[partition_slot - 1]
    p_end = partition_end_arr[partition_slot]
    if tid >= p_end - p_start:
        return

    ci = partition_data[p_start + tid]

    n = c_normal[ci]
    t1 = compute_tangent_frame(n)
    t2 = wp.cross(t1, n)
    c_tangent1[ci] = t1

    b0 = c_body0[ci]
    b1 = c_body1[ci]

    q0 = b_orientation[b0]
    q1 = b_orientation[b1]

    rw0 = wp.quat_rotate(q0, c_offset0[ci])
    rw1 = wp.quat_rotate(q1, c_offset1[ci])
    c_rel0[ci] = rw0
    c_rel1[ci] = rw1

    pos0 = b_position[b0]
    pos1 = b_position[b1]

    # Newton convention: offsets point to margin-inward reference
    # points; subtract margin0 + margin1 to get actual surface gap.
    thickness = c_margin0[ci] + c_margin1[ci]
    gap = wp.dot(n, (pos1 + rw1) - (pos0 + rw0)) - thickness
    c_bias[ci] = -compute_contact_bias(-gap, inv_dt)

    inv_m0 = b_inverse_mass[b0]
    inv_m1 = b_inverse_mass[b1]
    inv_i0 = b_inverse_inertia_world[b0]
    inv_i1 = b_inverse_inertia_world[b1]

    # Mass splitting: each body's inverse-mass contribution is scaled
    # by the number of contacts on that body.  Static bodies (count 0
    # or mass 0) get split factor 1 so they remain zero.
    nc0 = contact_count_per_body[b0]
    nc1 = contact_count_per_body[b1]
    split0 = wp.max(float(nc0), 1.0)
    split1 = wp.max(float(nc1), 1.0)

    c_eff_n[ci] = compute_effective_mass_split(inv_m0, inv_m1, inv_i0, inv_i1, rw0, rw1, n, split0, split1)
    c_eff_t1[ci] = compute_effective_mass_split(inv_m0, inv_m1, inv_i0, inv_i1, rw0, rw1, t1, split0, split1)
    c_eff_t2[ci] = compute_effective_mass_split(inv_m0, inv_m1, inv_i0, inv_i1, rw0, rw1, t2, split0, split1)

    # Warm start: apply accumulated impulses with FULL (unsplit) mass,
    # scaled by ImpulseInheritanceFactor (0.90) matching C# PhoenX.
    acc_n = c_accumulated_n[ci] * WARM_START_SCALE
    acc_t1 = c_accumulated_t1[ci] * WARM_START_SCALE
    acc_t2 = c_accumulated_t2[ci] * WARM_START_SCALE
    c_accumulated_n[ci] = acc_n
    c_accumulated_t1[ci] = acc_t1
    c_accumulated_t2[ci] = acc_t2
    impulse = acc_n * n + acc_t1 * t1 + acc_t2 * t2

    is_static_0 = (b_flags[b0] & BODY_FLAG_STATIC) != 0 or inv_m0 == 0.0
    is_static_1 = (b_flags[b1] & BODY_FLAG_STATIC) != 0 or inv_m1 == 0.0

    if not is_static_0:
        b_velocity[b0] = b_velocity[b0] - inv_m0 * impulse
        b_angular_velocity[b0] = b_angular_velocity[b0] - inv_i0 * wp.cross(rw0, impulse)

    if not is_static_1:
        b_velocity[b1] = b_velocity[b1] + inv_m1 * impulse
        b_angular_velocity[b1] = b_angular_velocity[b1] + inv_i1 * wp.cross(rw1, impulse)


# ---------------------------------------------------------------------------
# PGS iteration kernel
# ---------------------------------------------------------------------------


@wp.kernel
def solve_contacts_kernel(
    partition_data: wp.array(dtype=wp.int32),
    partition_end_arr: wp.array(dtype=wp.int32),
    partition_slot: int,
    c_normal: wp.array(dtype=wp.vec3),
    c_tangent1: wp.array(dtype=wp.vec3),
    c_body0: wp.array(dtype=wp.int32),
    c_body1: wp.array(dtype=wp.int32),
    c_accumulated_n: wp.array(dtype=wp.float32),
    c_accumulated_t1: wp.array(dtype=wp.float32),
    c_accumulated_t2: wp.array(dtype=wp.float32),
    c_rel0: wp.array(dtype=wp.vec3),
    c_rel1: wp.array(dtype=wp.vec3),
    c_eff_n: wp.array(dtype=wp.float32),
    c_eff_t1: wp.array(dtype=wp.float32),
    c_eff_t2: wp.array(dtype=wp.float32),
    c_bias: wp.array(dtype=wp.float32),
    c_friction: wp.array(dtype=wp.float32),
    b_velocity: wp.array(dtype=wp.vec3),
    b_angular_velocity: wp.array(dtype=wp.vec3),
    b_inverse_mass: wp.array(dtype=wp.float32),
    b_inverse_inertia_world: wp.array(dtype=wp.mat33),
    b_flags: wp.array(dtype=wp.int32),
    use_bias: int,
):
    """One PGS iteration for a single partition's contacts."""
    tid = wp.tid()
    p_start = int(0)
    if partition_slot > 0:
        p_start = partition_end_arr[partition_slot - 1]
    p_end = partition_end_arr[partition_slot]
    if tid >= p_end - p_start:
        return

    ci = partition_data[p_start + tid]

    n = c_normal[ci]
    t1 = c_tangent1[ci]
    t2 = wp.cross(t1, n)

    b0 = c_body0[ci]
    b1 = c_body1[ci]

    rw0 = c_rel0[ci]
    rw1 = c_rel1[ci]

    v0 = b_velocity[b0]
    w0 = b_angular_velocity[b0]
    v1 = b_velocity[b1]
    w1 = b_angular_velocity[b1]

    # Relative velocity at contact point: v_rel = (v1 + w1 x r1) - (v0 + w0 x r0)
    dv = (v1 + wp.cross(w1, rw1)) - (v0 + wp.cross(w0, rw0))

    dv_n = wp.dot(n, dv)
    dv_t1 = wp.dot(t1, dv)
    dv_t2 = wp.dot(t2, dv)

    bias_val = 0.0
    if use_bias != 0:
        bias_val = c_bias[ci]

    eff_n = c_eff_n[ci]
    eff_t1 = c_eff_t1[ci]
    eff_t2 = c_eff_t2[ci]

    # Normal impulse correction: lambda_n = -(Jv + bias) * W_n, clamped >= 0
    delta_n = -(dv_n + bias_val) * eff_n
    old_acc_n = c_accumulated_n[ci]
    new_acc_n = wp.max(old_acc_n + delta_n, 0.0)
    applied_n = new_acc_n - old_acc_n
    c_accumulated_n[ci] = new_acc_n

    # Friction impulse: clamped within [-mu * lambda_n, +mu * lambda_n]
    mu = c_friction[ci]
    max_friction = mu * new_acc_n

    delta_t1 = -dv_t1 * eff_t1
    old_acc_t1 = c_accumulated_t1[ci]
    new_acc_t1 = wp.clamp(old_acc_t1 + delta_t1, -max_friction, max_friction)
    applied_t1 = new_acc_t1 - old_acc_t1
    c_accumulated_t1[ci] = new_acc_t1

    delta_t2 = -dv_t2 * eff_t2
    old_acc_t2 = c_accumulated_t2[ci]
    new_acc_t2 = wp.clamp(old_acc_t2 + delta_t2, -max_friction, max_friction)
    applied_t2 = new_acc_t2 - old_acc_t2
    c_accumulated_t2[ci] = new_acc_t2

    # Convert to world impulse and apply to bodies
    impulse = applied_n * n + applied_t1 * t1 + applied_t2 * t2

    inv_m0 = b_inverse_mass[b0]
    inv_m1 = b_inverse_mass[b1]
    inv_i0 = b_inverse_inertia_world[b0]
    inv_i1 = b_inverse_inertia_world[b1]

    is_static_0 = (b_flags[b0] & BODY_FLAG_STATIC) != 0 or inv_m0 == 0.0
    is_static_1 = (b_flags[b1] & BODY_FLAG_STATIC) != 0 or inv_m1 == 0.0

    if not is_static_0:
        b_velocity[b0] = v0 - inv_m0 * impulse
        b_angular_velocity[b0] = w0 - inv_i0 * wp.cross(rw0, impulse)

    if not is_static_1:
        b_velocity[b1] = v1 + inv_m1 * impulse
        b_angular_velocity[b1] = w1 + inv_i1 * wp.cross(rw1, impulse)


# ---------------------------------------------------------------------------
# Contact visualization
# ---------------------------------------------------------------------------

CONTACT_LINE_LENGTH = 0.1


@wp.kernel
def build_contact_lines_kernel(
    c_body0: wp.array(dtype=wp.int32),
    c_offset0: wp.array(dtype=wp.vec3),
    c_normal: wp.array(dtype=wp.vec3),
    b_position: wp.array(dtype=wp.vec3),
    b_orientation: wp.array(dtype=wp.quat),
    count: wp.array(dtype=wp.int32),
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
):
    """Build world-space line segments for contact visualization."""
    tid = wp.tid()
    if tid >= count[0]:
        line_starts[tid] = wp.vec3(0.0, 0.0, 0.0)
        line_ends[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    b0 = c_body0[tid]
    p = b_position[b0]
    q = b_orientation[b0]
    world_pt = p + wp.quat_rotate(q, c_offset0[tid])
    n = c_normal[tid]
    line_starts[tid] = world_pt
    line_ends[tid] = world_pt + n * CONTACT_LINE_LENGTH


# ---------------------------------------------------------------------------
# ContactKernels — bakes DataStore column offsets via wp.static()
# ---------------------------------------------------------------------------


class ContactKernels:
    """Compiled contact kernels bound to specific contact and body stores.

    Each kernel receives only two flat ``float32`` arrays (the backing buffers
    of the contact :class:`DataStore` and body :class:`HandleStore`) plus
    partition metadata and the per-body contact count array.  Column offsets
    are baked as compile-time integer constants via ``wp.static(col_base(...))``.

    Args:
        contact_store: :class:`DataStore` for :class:`ContactPointSchema`.
        body_store: :class:`HandleStore` for :class:`RigidBodySchema`.
    """

    def __init__(self, contact_store, body_store):
        body_ds = body_store.store  # HandleStore wraps a DataStore

        # Contact column base indices (baked via wp.static in kernels)
        c_body0 = col_base(contact_store, "body0")
        c_body1 = col_base(contact_store, "body1")
        c_normal = col_base(contact_store, "normal")
        c_offset0 = col_base(contact_store, "offset0")
        c_offset1 = col_base(contact_store, "offset1")
        c_accumulated_n = col_base(contact_store, "accumulated_normal_impulse")
        c_accumulated_t1 = col_base(contact_store, "accumulated_tangent_impulse1")
        c_accumulated_t2 = col_base(contact_store, "accumulated_tangent_impulse2")
        c_friction = col_base(contact_store, "friction")
        c_tangent1 = col_base(contact_store, "tangent1")
        c_rel0 = col_base(contact_store, "rel_pos_world0")
        c_rel1 = col_base(contact_store, "rel_pos_world1")
        c_eff_n = col_base(contact_store, "effective_mass_n")
        c_eff_t1 = col_base(contact_store, "effective_mass_t1")
        c_eff_t2 = col_base(contact_store, "effective_mass_t2")
        c_bias = col_base(contact_store, "bias")
        c_margin0 = col_base(contact_store, "margin0")
        c_margin1 = col_base(contact_store, "margin1")

        # Body column base indices
        b_position = col_base(body_ds, "position")
        b_orientation = col_base(body_ds, "orientation")
        b_velocity = col_base(body_ds, "velocity")
        b_angular_velocity = col_base(body_ds, "angular_velocity")
        b_inverse_mass = col_base(body_ds, "inverse_mass")
        b_inverse_inertia_world = col_base(body_ds, "inverse_inertia_world")
        b_flags = col_base(body_ds, "flags")

        # ---------------------------------------------------------------
        # Prepare kernel
        # ---------------------------------------------------------------

        @wp.kernel
        def _prepare(
            cdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            partition_data: wp.array(dtype=wp.int32),
            partition_ends: wp.array(dtype=wp.int32),
            partition_slot: int,
            contact_count_per_body: wp.array(dtype=wp.int32),
            inv_dt: float,
        ):
            """Compute per-contact solver data and apply warm-start impulses.

            Effective masses use split inverse masses (Tonge et al. 2012)
            for faster PGS convergence.  Warm-start impulses use the full
            inverse masses so that momentum is conserved.
            """
            tid = wp.tid()
            p_start = int(0)
            if partition_slot > 0:
                p_start = partition_ends[partition_slot - 1]
            p_end = partition_ends[partition_slot]
            if tid >= p_end - p_start:
                return

            ci = partition_data[p_start + tid]

            n = ds_load_vec3(cdata, wp.static(c_normal), ci)
            t1 = compute_tangent_frame(n)
            t2 = wp.cross(t1, n)
            ds_store_vec3(cdata, wp.static(c_tangent1), ci, t1)

            b0 = ds_load_int(cdata, wp.static(c_body0), ci)
            b1 = ds_load_int(cdata, wp.static(c_body1), ci)

            q0 = ds_load_quat(bdata, wp.static(b_orientation), b0)
            q1 = ds_load_quat(bdata, wp.static(b_orientation), b1)

            rw0 = wp.quat_rotate(q0, ds_load_vec3(cdata, wp.static(c_offset0), ci))
            rw1 = wp.quat_rotate(q1, ds_load_vec3(cdata, wp.static(c_offset1), ci))
            ds_store_vec3(cdata, wp.static(c_rel0), ci, rw0)
            ds_store_vec3(cdata, wp.static(c_rel1), ci, rw1)

            pos0 = ds_load_vec3(bdata, wp.static(b_position), b0)
            pos1 = ds_load_vec3(bdata, wp.static(b_position), b1)

            # Newton convention: offsets point to margin-inward reference
            # points; subtract margin0 + margin1 to get actual surface gap.
            thickness = ds_load_float(cdata, wp.static(c_margin0), ci) + ds_load_float(cdata, wp.static(c_margin1), ci)
            gap = wp.dot(n, (pos1 + rw1) - (pos0 + rw0)) - thickness
            ds_store_float(cdata, wp.static(c_bias), ci, -compute_contact_bias(-gap, inv_dt))

            inv_m0 = ds_load_float(bdata, wp.static(b_inverse_mass), b0)
            inv_m1 = ds_load_float(bdata, wp.static(b_inverse_mass), b1)
            inv_i0 = ds_load_mat33(bdata, wp.static(b_inverse_inertia_world), b0)
            inv_i1 = ds_load_mat33(bdata, wp.static(b_inverse_inertia_world), b1)

            # Mass splitting: each body's inverse-mass contribution is scaled
            # by the number of contacts on that body.  Static bodies (count 0
            # or mass 0) get split factor 1 so they remain zero.
            nc0 = contact_count_per_body[b0]
            nc1 = contact_count_per_body[b1]
            split0 = wp.max(float(nc0), 1.0)
            split1 = wp.max(float(nc1), 1.0)

            ds_store_float(
                cdata,
                wp.static(c_eff_n),
                ci,
                compute_effective_mass_split(inv_m0, inv_m1, inv_i0, inv_i1, rw0, rw1, n, split0, split1),
            )
            ds_store_float(
                cdata,
                wp.static(c_eff_t1),
                ci,
                compute_effective_mass_split(inv_m0, inv_m1, inv_i0, inv_i1, rw0, rw1, t1, split0, split1),
            )
            ds_store_float(
                cdata,
                wp.static(c_eff_t2),
                ci,
                compute_effective_mass_split(inv_m0, inv_m1, inv_i0, inv_i1, rw0, rw1, t2, split0, split1),
            )

            # Warm start: apply accumulated impulses with FULL (unsplit) mass,
            # scaled by ImpulseInheritanceFactor (0.90) matching C# PhoenX.
            acc_n = ds_load_float(cdata, wp.static(c_accumulated_n), ci) * WARM_START_SCALE
            acc_t1 = ds_load_float(cdata, wp.static(c_accumulated_t1), ci) * WARM_START_SCALE
            acc_t2 = ds_load_float(cdata, wp.static(c_accumulated_t2), ci) * WARM_START_SCALE
            ds_store_float(cdata, wp.static(c_accumulated_n), ci, acc_n)
            ds_store_float(cdata, wp.static(c_accumulated_t1), ci, acc_t1)
            ds_store_float(cdata, wp.static(c_accumulated_t2), ci, acc_t2)
            impulse = acc_n * n + acc_t1 * t1 + acc_t2 * t2

            f0 = ds_load_int(bdata, wp.static(b_flags), b0)
            f1 = ds_load_int(bdata, wp.static(b_flags), b1)
            is_static_0 = (f0 & BODY_FLAG_STATIC) != 0 or inv_m0 == 0.0
            is_static_1 = (f1 & BODY_FLAG_STATIC) != 0 or inv_m1 == 0.0

            if not is_static_0:
                v0 = ds_load_vec3(bdata, wp.static(b_velocity), b0)
                w0 = ds_load_vec3(bdata, wp.static(b_angular_velocity), b0)
                ds_store_vec3(bdata, wp.static(b_velocity), b0, v0 - inv_m0 * impulse)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b0, w0 - inv_i0 * wp.cross(rw0, impulse))

            if not is_static_1:
                v1 = ds_load_vec3(bdata, wp.static(b_velocity), b1)
                w1 = ds_load_vec3(bdata, wp.static(b_angular_velocity), b1)
                ds_store_vec3(bdata, wp.static(b_velocity), b1, v1 + inv_m1 * impulse)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b1, w1 + inv_i1 * wp.cross(rw1, impulse))

        # ---------------------------------------------------------------
        # Solve kernel
        # ---------------------------------------------------------------

        @wp.kernel
        def _solve(
            cdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            partition_data: wp.array(dtype=wp.int32),
            partition_ends: wp.array(dtype=wp.int32),
            partition_slot: int,
            use_bias: int,
        ):
            """One PGS iteration for a single partition's contacts."""
            tid = wp.tid()
            p_start = int(0)
            if partition_slot > 0:
                p_start = partition_ends[partition_slot - 1]
            p_end = partition_ends[partition_slot]
            if tid >= p_end - p_start:
                return

            ci = partition_data[p_start + tid]

            n = ds_load_vec3(cdata, wp.static(c_normal), ci)
            t1 = ds_load_vec3(cdata, wp.static(c_tangent1), ci)
            t2 = wp.cross(t1, n)

            b0 = ds_load_int(cdata, wp.static(c_body0), ci)
            b1 = ds_load_int(cdata, wp.static(c_body1), ci)

            rw0 = ds_load_vec3(cdata, wp.static(c_rel0), ci)
            rw1 = ds_load_vec3(cdata, wp.static(c_rel1), ci)

            v0 = ds_load_vec3(bdata, wp.static(b_velocity), b0)
            w0 = ds_load_vec3(bdata, wp.static(b_angular_velocity), b0)
            v1 = ds_load_vec3(bdata, wp.static(b_velocity), b1)
            w1 = ds_load_vec3(bdata, wp.static(b_angular_velocity), b1)

            # Relative velocity at contact point: v_rel = (v1 + w1 x r1) - (v0 + w0 x r0)
            dv = (v1 + wp.cross(w1, rw1)) - (v0 + wp.cross(w0, rw0))

            dv_n = wp.dot(n, dv)
            dv_t1 = wp.dot(t1, dv)
            dv_t2 = wp.dot(t2, dv)

            bias_val = 0.0
            if use_bias != 0:
                bias_val = ds_load_float(cdata, wp.static(c_bias), ci)

            eff_n = ds_load_float(cdata, wp.static(c_eff_n), ci)
            eff_t1 = ds_load_float(cdata, wp.static(c_eff_t1), ci)
            eff_t2 = ds_load_float(cdata, wp.static(c_eff_t2), ci)

            # Normal impulse correction: lambda_n = -(Jv + bias) * W_n, clamped >= 0
            delta_n = -(dv_n + bias_val) * eff_n
            old_acc_n = ds_load_float(cdata, wp.static(c_accumulated_n), ci)
            new_acc_n = wp.max(old_acc_n + delta_n, 0.0)
            applied_n = new_acc_n - old_acc_n
            ds_store_float(cdata, wp.static(c_accumulated_n), ci, new_acc_n)

            # Friction impulse: clamped within [-mu * lambda_n, +mu * lambda_n]
            mu = ds_load_float(cdata, wp.static(c_friction), ci)
            max_friction = mu * new_acc_n

            delta_t1 = -dv_t1 * eff_t1
            old_acc_t1 = ds_load_float(cdata, wp.static(c_accumulated_t1), ci)
            new_acc_t1 = wp.clamp(old_acc_t1 + delta_t1, -max_friction, max_friction)
            applied_t1 = new_acc_t1 - old_acc_t1
            ds_store_float(cdata, wp.static(c_accumulated_t1), ci, new_acc_t1)

            delta_t2 = -dv_t2 * eff_t2
            old_acc_t2 = ds_load_float(cdata, wp.static(c_accumulated_t2), ci)
            new_acc_t2 = wp.clamp(old_acc_t2 + delta_t2, -max_friction, max_friction)
            applied_t2 = new_acc_t2 - old_acc_t2
            ds_store_float(cdata, wp.static(c_accumulated_t2), ci, new_acc_t2)

            # Convert to world impulse and apply to bodies
            impulse = applied_n * n + applied_t1 * t1 + applied_t2 * t2

            inv_m0 = ds_load_float(bdata, wp.static(b_inverse_mass), b0)
            inv_m1 = ds_load_float(bdata, wp.static(b_inverse_mass), b1)
            inv_i0 = ds_load_mat33(bdata, wp.static(b_inverse_inertia_world), b0)
            inv_i1 = ds_load_mat33(bdata, wp.static(b_inverse_inertia_world), b1)

            f0 = ds_load_int(bdata, wp.static(b_flags), b0)
            f1 = ds_load_int(bdata, wp.static(b_flags), b1)
            is_static_0 = (f0 & BODY_FLAG_STATIC) != 0 or inv_m0 == 0.0
            is_static_1 = (f1 & BODY_FLAG_STATIC) != 0 or inv_m1 == 0.0

            if not is_static_0:
                ds_store_vec3(bdata, wp.static(b_velocity), b0, v0 - inv_m0 * impulse)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b0, w0 - inv_i0 * wp.cross(rw0, impulse))

            if not is_static_1:
                ds_store_vec3(bdata, wp.static(b_velocity), b1, v1 + inv_m1 * impulse)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b1, w1 + inv_i1 * wp.cross(rw1, impulse))

        # ---------------------------------------------------------------
        # Bundled prepare kernel (one thread per bundle)
        # ---------------------------------------------------------------

        @wp.kernel
        def _prepare_bundled(
            cdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            partition_data: wp.array(dtype=wp.int32),
            partition_ends: wp.array(dtype=wp.int32),
            partition_slot: int,
            bundle_starts: wp.array(dtype=wp.int32),
            bundle_count: wp.array(dtype=wp.int32),
            sort_perm: wp.array(dtype=wp.int32),
            contact_count_per_body: wp.array(dtype=wp.int32),
            inv_dt: float,
        ):
            """Compute per-contact solver data and warm-start for a bundle of contacts."""
            tid = wp.tid()
            p_start = int(0)
            if partition_slot > 0:
                p_start = partition_ends[partition_slot - 1]
            p_end = partition_ends[partition_slot]
            if tid >= p_end - p_start:
                return

            bundle_idx = partition_data[p_start + tid]
            if bundle_idx >= bundle_count[0]:
                return  # joint element, skip

            b_start = bundle_starts[bundle_idx]
            b_end = bundle_starts[bundle_idx + 1]

            # Load body state from first contact in bundle
            first_ci = sort_perm[b_start]
            b0 = ds_load_int(cdata, wp.static(c_body0), first_ci)
            b1 = ds_load_int(cdata, wp.static(c_body1), first_ci)

            q0 = ds_load_quat(bdata, wp.static(b_orientation), b0)
            q1 = ds_load_quat(bdata, wp.static(b_orientation), b1)
            pos0 = ds_load_vec3(bdata, wp.static(b_position), b0)
            pos1 = ds_load_vec3(bdata, wp.static(b_position), b1)
            inv_m0 = ds_load_float(bdata, wp.static(b_inverse_mass), b0)
            inv_m1 = ds_load_float(bdata, wp.static(b_inverse_mass), b1)
            inv_i0 = ds_load_mat33(bdata, wp.static(b_inverse_inertia_world), b0)
            inv_i1 = ds_load_mat33(bdata, wp.static(b_inverse_inertia_world), b1)
            f0 = ds_load_int(bdata, wp.static(b_flags), b0)
            f1 = ds_load_int(bdata, wp.static(b_flags), b1)
            is_static_0 = (f0 & BODY_FLAG_STATIC) != 0 or inv_m0 == 0.0
            is_static_1 = (f1 & BODY_FLAG_STATIC) != 0 or inv_m1 == 0.0

            nc0 = contact_count_per_body[b0]
            nc1 = contact_count_per_body[b1]
            split0 = wp.max(float(nc0), 1.0)
            split1 = wp.max(float(nc1), 1.0)

            v0 = ds_load_vec3(bdata, wp.static(b_velocity), b0)
            w0 = ds_load_vec3(bdata, wp.static(b_angular_velocity), b0)
            v1 = ds_load_vec3(bdata, wp.static(b_velocity), b1)
            w1 = ds_load_vec3(bdata, wp.static(b_angular_velocity), b1)

            for s in range(b_start, b_end):
                ci = sort_perm[s]

                n = ds_load_vec3(cdata, wp.static(c_normal), ci)
                t1 = compute_tangent_frame(n)
                t2 = wp.cross(t1, n)
                ds_store_vec3(cdata, wp.static(c_tangent1), ci, t1)

                rw0 = wp.quat_rotate(q0, ds_load_vec3(cdata, wp.static(c_offset0), ci))
                rw1 = wp.quat_rotate(q1, ds_load_vec3(cdata, wp.static(c_offset1), ci))
                ds_store_vec3(cdata, wp.static(c_rel0), ci, rw0)
                ds_store_vec3(cdata, wp.static(c_rel1), ci, rw1)

                thickness = ds_load_float(cdata, wp.static(c_margin0), ci) + ds_load_float(
                    cdata, wp.static(c_margin1), ci
                )
                gap = wp.dot(n, (pos1 + rw1) - (pos0 + rw0)) - thickness
                ds_store_float(cdata, wp.static(c_bias), ci, -compute_contact_bias(-gap, inv_dt))

                ds_store_float(
                    cdata,
                    wp.static(c_eff_n),
                    ci,
                    compute_effective_mass_split(inv_m0, inv_m1, inv_i0, inv_i1, rw0, rw1, n, split0, split1),
                )
                ds_store_float(
                    cdata,
                    wp.static(c_eff_t1),
                    ci,
                    compute_effective_mass_split(inv_m0, inv_m1, inv_i0, inv_i1, rw0, rw1, t1, split0, split1),
                )
                ds_store_float(
                    cdata,
                    wp.static(c_eff_t2),
                    ci,
                    compute_effective_mass_split(inv_m0, inv_m1, inv_i0, inv_i1, rw0, rw1, t2, split0, split1),
                )

                acc_n = ds_load_float(cdata, wp.static(c_accumulated_n), ci) * WARM_START_SCALE
                acc_t1 = ds_load_float(cdata, wp.static(c_accumulated_t1), ci) * WARM_START_SCALE
                acc_t2 = ds_load_float(cdata, wp.static(c_accumulated_t2), ci) * WARM_START_SCALE
                ds_store_float(cdata, wp.static(c_accumulated_n), ci, acc_n)
                ds_store_float(cdata, wp.static(c_accumulated_t1), ci, acc_t1)
                ds_store_float(cdata, wp.static(c_accumulated_t2), ci, acc_t2)
                impulse = acc_n * n + acc_t1 * t1 + acc_t2 * t2

                if not is_static_0:
                    v0 = v0 - inv_m0 * impulse
                    w0 = w0 - inv_i0 * wp.cross(rw0, impulse)
                if not is_static_1:
                    v1 = v1 + inv_m1 * impulse
                    w1 = w1 + inv_i1 * wp.cross(rw1, impulse)

            if not is_static_0:
                ds_store_vec3(bdata, wp.static(b_velocity), b0, v0)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b0, w0)
            if not is_static_1:
                ds_store_vec3(bdata, wp.static(b_velocity), b1, v1)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b1, w1)

        # ---------------------------------------------------------------
        # Bundled solve kernel (one thread per bundle)
        # ---------------------------------------------------------------

        @wp.kernel
        def _solve_bundled(
            cdata: wp.array(dtype=wp.float32),
            bdata: wp.array(dtype=wp.float32),
            partition_data: wp.array(dtype=wp.int32),
            partition_ends: wp.array(dtype=wp.int32),
            partition_slot: int,
            bundle_starts: wp.array(dtype=wp.int32),
            bundle_count: wp.array(dtype=wp.int32),
            sort_perm: wp.array(dtype=wp.int32),
            use_bias: int,
        ):
            """One PGS iteration for a single partition's contact bundles."""
            tid = wp.tid()
            p_start = int(0)
            if partition_slot > 0:
                p_start = partition_ends[partition_slot - 1]
            p_end = partition_ends[partition_slot]
            if tid >= p_end - p_start:
                return

            bundle_idx = partition_data[p_start + tid]
            if bundle_idx >= bundle_count[0]:
                return  # joint element, skip

            b_start = bundle_starts[bundle_idx]
            b_end = bundle_starts[bundle_idx + 1]

            first_ci = sort_perm[b_start]
            b0 = ds_load_int(cdata, wp.static(c_body0), first_ci)
            b1 = ds_load_int(cdata, wp.static(c_body1), first_ci)

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

            for s in range(b_start, b_end):
                ci = sort_perm[s]

                n = ds_load_vec3(cdata, wp.static(c_normal), ci)
                t1 = ds_load_vec3(cdata, wp.static(c_tangent1), ci)
                t2 = wp.cross(t1, n)

                rw0 = ds_load_vec3(cdata, wp.static(c_rel0), ci)
                rw1 = ds_load_vec3(cdata, wp.static(c_rel1), ci)

                dv = (v1 + wp.cross(w1, rw1)) - (v0 + wp.cross(w0, rw0))

                dv_n = wp.dot(n, dv)
                dv_t1 = wp.dot(t1, dv)
                dv_t2 = wp.dot(t2, dv)

                bias_val = 0.0
                if use_bias != 0:
                    bias_val = ds_load_float(cdata, wp.static(c_bias), ci)

                eff_n = ds_load_float(cdata, wp.static(c_eff_n), ci)
                eff_t1 = ds_load_float(cdata, wp.static(c_eff_t1), ci)
                eff_t2 = ds_load_float(cdata, wp.static(c_eff_t2), ci)

                delta_n = -(dv_n + bias_val) * eff_n
                old_acc_n = ds_load_float(cdata, wp.static(c_accumulated_n), ci)
                new_acc_n = wp.max(old_acc_n + delta_n, 0.0)
                applied_n = new_acc_n - old_acc_n
                ds_store_float(cdata, wp.static(c_accumulated_n), ci, new_acc_n)

                mu = ds_load_float(cdata, wp.static(c_friction), ci)
                max_friction = mu * new_acc_n

                delta_t1 = -dv_t1 * eff_t1
                old_acc_t1 = ds_load_float(cdata, wp.static(c_accumulated_t1), ci)
                new_acc_t1 = wp.clamp(old_acc_t1 + delta_t1, -max_friction, max_friction)
                applied_t1 = new_acc_t1 - old_acc_t1
                ds_store_float(cdata, wp.static(c_accumulated_t1), ci, new_acc_t1)

                delta_t2 = -dv_t2 * eff_t2
                old_acc_t2 = ds_load_float(cdata, wp.static(c_accumulated_t2), ci)
                new_acc_t2 = wp.clamp(old_acc_t2 + delta_t2, -max_friction, max_friction)
                applied_t2 = new_acc_t2 - old_acc_t2
                ds_store_float(cdata, wp.static(c_accumulated_t2), ci, new_acc_t2)

                impulse = applied_n * n + applied_t1 * t1 + applied_t2 * t2

                if not is_static_0:
                    v0 = v0 - inv_m0 * impulse
                    w0 = w0 - inv_i0 * wp.cross(rw0, impulse)
                if not is_static_1:
                    v1 = v1 + inv_m1 * impulse
                    w1 = w1 + inv_i1 * wp.cross(rw1, impulse)

            if not is_static_0:
                ds_store_vec3(bdata, wp.static(b_velocity), b0, v0)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b0, w0)
            if not is_static_1:
                ds_store_vec3(bdata, wp.static(b_velocity), b1, v1)
                ds_store_vec3(bdata, wp.static(b_angular_velocity), b1, w1)

        # Store compiled kernels and data arrays for launch
        self.prepare = _prepare
        self.solve = _solve
        self.prepare_bundled = _prepare_bundled
        self.solve_bundled = _solve_bundled
        self.contact_data = contact_store.data
        self.body_data = body_ds.data
        self.device = contact_store.device
        self.contact_capacity = contact_store.capacity
