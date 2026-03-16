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

"""PhoenX XPBD position-level contact constraint kernels.

Alternative to velocity-level PGS contacts in :mod:`contacts`. Position-level
solving resolves penetration directly by modifying body positions and orientations,
then derives velocities from the resulting displacement.
"""

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
from .contacts import compute_effective_mass_split, compute_tangent_frame
from .schemas import BODY_FLAG_STATIC

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARM_START_SCALE = wp.constant(0.90)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


@wp.func
def _ds_store_quat(store: wp.array(dtype=wp.float32), base: int, row: int, q: wp.quat):
    """Store a quaternion into the flat DataStore buffer."""
    i = base + row * 4
    store[i] = q[0]
    store[i + 1] = q[1]
    store[i + 2] = q[2]
    store[i + 3] = q[3]


@wp.func
def _quat_apply_angular_delta(q: wp.quat, w: wp.vec3) -> wp.quat:
    """Apply a finite angular correction to a quaternion."""
    angle = wp.length(w)
    if angle > 1.0e-8:
        axis = wp.normalize(w)
        dq = wp.quat_from_axis_angle(axis, angle)
        return wp.normalize(dq * q)
    return q


# ---------------------------------------------------------------------------
# Standalone helper kernels
# ---------------------------------------------------------------------------


@wp.kernel
def store_ref_state_kernel(
    position: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    ref_position: wp.array(dtype=wp.vec3),
    ref_orientation: wp.array(dtype=wp.quat),
    count: wp.array(dtype=wp.int32),
):
    """Copy current position/orientation to reference arrays before solving."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    ref_position[tid] = position[tid]
    ref_orientation[tid] = orientation[tid]


@wp.kernel
def predict_positions_kernel(
    position: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    velocity: wp.array(dtype=wp.vec3),
    angular_velocity: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
    dt: float,
    count: wp.array(dtype=wp.int32),
):
    """Forward-integrate positions from velocities."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    if (flags[tid] & BODY_FLAG_STATIC) != 0:
        return

    pos = position[tid]
    orient = orientation[tid]
    vel = velocity[tid]
    angvel = angular_velocity[tid]

    position[tid] = pos + vel * dt

    # Quaternion integration: q' = q + 0.5 * dt * [wx, wy, wz, 0] * q
    half_dt = 0.5 * dt
    dq = wp.quat(
        half_dt * (angvel[0] * orient[3] + angvel[1] * orient[2] - angvel[2] * orient[1]),
        half_dt * (-angvel[0] * orient[2] + angvel[1] * orient[3] + angvel[2] * orient[0]),
        half_dt * (angvel[0] * orient[1] - angvel[1] * orient[0] + angvel[2] * orient[3]),
        half_dt * (-angvel[0] * orient[0] - angvel[1] * orient[1] - angvel[2] * orient[2]),
    )
    orientation[tid] = wp.normalize(
        wp.quat(
            orient[0] + dq[0],
            orient[1] + dq[1],
            orient[2] + dq[2],
            orient[3] + dq[3],
        )
    )


@wp.kernel
def derive_velocities_kernel(
    position: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    ref_position: wp.array(dtype=wp.vec3),
    ref_orientation: wp.array(dtype=wp.quat),
    velocity: wp.array(dtype=wp.vec3),
    angular_velocity: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.int32),
    inv_dt: float,
    count: wp.array(dtype=wp.int32),
):
    """Derive velocities from position change: vel = (pos - ref_pos) * inv_dt."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    if (flags[tid] & BODY_FLAG_STATIC) != 0:
        return

    cur_pos = position[tid]
    ref_pos = ref_position[tid]
    velocity[tid] = (cur_pos - ref_pos) * inv_dt

    cur_orient = orientation[tid]
    ref_orient = ref_orientation[tid]
    delta_q = cur_orient * wp.quat_inverse(ref_orient)
    qv = wp.vec3(delta_q[0], delta_q[1], delta_q[2])
    angvel = 2.0 * inv_dt * qv
    if delta_q[3] < 0.0:
        angvel = -angvel
    angular_velocity[tid] = angvel


# ---------------------------------------------------------------------------
# ContactKernelsXPBD
# ---------------------------------------------------------------------------


class ContactKernelsXPBD:
    """Compiled XPBD contact kernels bound to specific contact and body stores.

    Position-level alternative to :class:`~contacts.ContactKernels`.
    Same interface: prepare_bundled, solve_bundled.
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
        # Prepare kernel (identical to PGS version)
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

            b_start_idx = bundle_starts[bundle_idx]
            b_end_idx = bundle_starts[bundle_idx + 1]

            # Load body state from first contact in bundle
            first_ci = sort_perm[b_start_idx]
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

            for s in range(b_start_idx, b_end_idx):
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
                ds_store_float(cdata, wp.static(c_bias), ci, gap)

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
        # Solve kernel (position-level XPBD)
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
            """One XPBD iteration for a single partition's contact bundles.

            Resolves penetration by directly modifying body positions and
            orientations rather than velocities.  The *use_bias* parameter
            is accepted for interface compatibility but ignored (XPBD does
            not use Baumgarte bias).
            """
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

            b_start_idx = bundle_starts[bundle_idx]
            b_end_idx = bundle_starts[bundle_idx + 1]

            first_ci = sort_perm[b_start_idx]
            b0 = ds_load_int(cdata, wp.static(c_body0), first_ci)
            b1 = ds_load_int(cdata, wp.static(c_body1), first_ci)

            # Load body positions and orientations (NOT velocities)
            pos0 = ds_load_vec3(bdata, wp.static(b_position), b0)
            pos1 = ds_load_vec3(bdata, wp.static(b_position), b1)
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

            for s in range(b_start_idx, b_end_idx):
                ci = sort_perm[s]

                n = ds_load_vec3(cdata, wp.static(c_normal), ci)
                t1 = ds_load_vec3(cdata, wp.static(c_tangent1), ci)
                t2 = wp.cross(t1, n)

                # Recompute world-space relative positions from body-local offsets
                rw0 = wp.quat_rotate(q0, ds_load_vec3(cdata, wp.static(c_offset0), ci))
                rw1 = wp.quat_rotate(q1, ds_load_vec3(cdata, wp.static(c_offset1), ci))

                # Compute gap
                thickness = ds_load_float(cdata, wp.static(c_margin0), ci) + ds_load_float(
                    cdata, wp.static(c_margin1), ci
                )
                sep = (pos1 + rw1) - (pos0 + rw0)
                gap = wp.dot(n, sep) - thickness

                if gap >= 0.0:
                    continue  # separated, skip

                eff_n = ds_load_float(cdata, wp.static(c_eff_n), ci)
                eff_t1 = ds_load_float(cdata, wp.static(c_eff_t1), ci)
                eff_t2 = ds_load_float(cdata, wp.static(c_eff_t2), ci)

                # Normal correction
                delta_n = -gap * eff_n
                old_acc_n = ds_load_float(cdata, wp.static(c_accumulated_n), ci)
                new_acc_n = wp.max(old_acc_n + delta_n, 0.0)
                applied_n = new_acc_n - old_acc_n
                ds_store_float(cdata, wp.static(c_accumulated_n), ci, new_acc_n)

                # Friction: tangential displacement
                mu = ds_load_float(cdata, wp.static(c_friction), ci)
                max_friction = mu * new_acc_n

                dt1 = wp.dot(t1, sep)
                delta_t1 = -dt1 * eff_t1
                old_acc_t1 = ds_load_float(cdata, wp.static(c_accumulated_t1), ci)
                new_acc_t1 = wp.clamp(old_acc_t1 + delta_t1, -max_friction, max_friction)
                applied_t1 = new_acc_t1 - old_acc_t1
                ds_store_float(cdata, wp.static(c_accumulated_t1), ci, new_acc_t1)

                dt2 = wp.dot(t2, sep)
                delta_t2 = -dt2 * eff_t2
                old_acc_t2 = ds_load_float(cdata, wp.static(c_accumulated_t2), ci)
                new_acc_t2 = wp.clamp(old_acc_t2 + delta_t2, -max_friction, max_friction)
                applied_t2 = new_acc_t2 - old_acc_t2
                ds_store_float(cdata, wp.static(c_accumulated_t2), ci, new_acc_t2)

                # Position correction vectors
                correction_n = applied_n * n
                correction_t = applied_t1 * t1 + applied_t2 * t2
                correction = correction_n + correction_t

                # Apply position and orientation corrections
                if not is_static_0:
                    pos0 = pos0 - inv_m0 * correction
                    ang_delta_0 = inv_i0 * wp.cross(rw0, correction)
                    q0 = _quat_apply_angular_delta(q0, -ang_delta_0)

                if not is_static_1:
                    pos1 = pos1 + inv_m1 * correction
                    ang_delta_1 = inv_i1 * wp.cross(rw1, correction)
                    q1 = _quat_apply_angular_delta(q1, ang_delta_1)

            # Write back final positions and orientations
            if not is_static_0:
                ds_store_vec3(bdata, wp.static(b_position), b0, pos0)
                _ds_store_quat(bdata, wp.static(b_orientation), b0, q0)
            if not is_static_1:
                ds_store_vec3(bdata, wp.static(b_position), b1, pos1)
                _ds_store_quat(bdata, wp.static(b_orientation), b1, q1)

        # Store compiled kernels and data arrays for launch
        self.prepare_bundled = _prepare_bundled
        self.solve_bundled = _solve_bundled
        self.contact_data = contact_store.data
        self.body_data = body_ds.data
        self.device = contact_store.device
        self.contact_capacity = contact_store.capacity
