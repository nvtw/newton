# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Temporal contact iterations: each biased PGS sweep of a substep is a contact substep.

Smaller contact steps can reduce oscillation when heavy bodies rest on a
light movable support, such as bottles in a carton. A full solver
substep reruns joint factorization, direct-contact responses, and row
preparation. Temporal iterations keep that work once per substep and advance
only the contact rows between sweeps:

* each sweep applies ``1/n`` of the substep's gravity and external forces
  (bodies on joints with free axes receive them at the substep start, see
  :func:`temporal_mark_jointed_components_kernel`),
* contact rows use the soft spring coefficients of ``dt/n`` and are warm
  started with the previous sweep's impulses, which are impulses over ``dt/n``,
* each row's gap advances by its relative normal motion over ``dt/n`` and its
  bias follows the gap, as when a substep reuses prepared rows,
* positions integrate the average of the sweeps' velocities, which is the
  motion the contact gaps tracked; bodies keep the last sweep's velocity.

Joint rows and exact joint projections run after every sweep as before. With
``n = 1`` nothing changes.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_solve import contact_normal_bias
from newton._src.solvers.phoenx.constraints.constraint_container import pd_coefficients
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_eff_n,
    cc_get_normal,
    cc_get_r0,
    cc_get_r1,
    cc_set_bias,
    cc_set_pd_bias,
    cc_set_pd_eff_soft,
    cc_set_pd_gamma,
)
from newton._src.solvers.phoenx.islands.island_builder import _find, _pack_entry, _unite
from newton._src.solvers.phoenx.solver_config import PHOENX_BOOST_CONTACT_NORMAL


@wp.kernel(enable_backward=False)
def temporal_contact_gap_kernel(
    columns: ContactColumnContainer,
    cc: ContactContainer,
    bodies: BodyContainer,
    column_count: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    rows_per_column: wp.int32,
    advance_dt: wp.float32,
    idt: wp.float32,
    bias_rate: wp.float32,
    column_stride: wp.int32,
    stiffness: wp.array[wp.float32],
    damping: wp.array[wp.float32],
):
    """Advance each ordinary contact row's gap by ``advance_dt`` of relative normal motion; refresh its bias.

    ``idt`` and ``bias_rate`` belong to the contact substep: a separated row admits
    the approach that closes its gap within one contact substep. Authored material
    coefficients use that same timestep and the updated penetration depth.
    """
    first_column, lane = wp.tid()
    for cid in range(first_column, column_count[0], column_stride):
        if owner[cid] >= wp.int32(0):
            continue
        b1 = contact_get_body1(columns, cid)
        b2 = contact_get_body2(columns, cid)
        v1 = bodies.velocity[b1]
        w1 = bodies.angular_velocity[b1]
        v2 = bodies.velocity[b2]
        w2 = bodies.angular_velocity[b2]
        first = contact_get_contact_first(columns, cid)
        count = contact_get_contact_count(columns, cid)
        for k in range(first + lane, first + count, rows_per_column):
            gap = cc.gap[k]
            if advance_dt > wp.float32(0.0):
                relative = v2 + wp.cross(w2, cc_get_r1(cc, k)) - v1 - wp.cross(w1, cc_get_r0(cc, k))
                gap = gap + advance_dt * wp.dot(relative, cc_get_normal(cc, k))
                cc.gap[k] = gap
            cc_set_bias(cc, k, contact_normal_bias(gap, idt, bias_rate))

            # Contact views and uncolored rows share canonical ordering, also
            # when compound-body grouping sorts the input material arrays.
            if stiffness.shape[0] > k or damping.shape[0] > k:
                k_n = wp.float32(0.0)
                c_n = wp.float32(0.0)
                if stiffness.shape[0] > k:
                    k_n = stiffness[k]
                if damping.shape[0] > k:
                    c_n = damping[k]
                eff_n = cc_get_eff_n(cc, k)
                gamma = wp.float32(0.0)
                bias = wp.float32(0.0)
                eff_soft = wp.float32(0.0)
                if gap <= wp.float32(0.0) and eff_n > wp.float32(0.0):
                    gamma, bias, eff_soft = pd_coefficients(
                        k_n,
                        c_n,
                        -gap,
                        wp.float32(1.0) / eff_n,
                        wp.float32(1.0) / idt,
                        PHOENX_BOOST_CONTACT_NORMAL,
                    )
                cc_set_pd_gamma(cc, k, gamma)
                cc_set_pd_bias(cc, k, bias)
                cc_set_pd_eff_soft(cc, k, eff_soft)


@wp.kernel(enable_backward=False)
def temporal_accumulate_velocity_kernel(
    bodies: BodyContainer,
    reset: wp.int32,
    velocity_sum: wp.array[wp.vec3f],
    angular_velocity_sum: wp.array[wp.vec3f],
):
    """Add every body's velocity to the substep's running sum (``reset`` starts a new sum)."""
    b = wp.tid()
    if reset != wp.int32(0):
        velocity_sum[b] = bodies.velocity[b]
        angular_velocity_sum[b] = bodies.angular_velocity[b]
    else:
        velocity_sum[b] = velocity_sum[b] + bodies.velocity[b]
        angular_velocity_sum[b] = angular_velocity_sum[b] + bodies.angular_velocity[b]


@wp.kernel(enable_backward=False)
def temporal_swap_average_velocity_kernel(
    bodies: BodyContainer,
    jointed: wp.array[wp.int32],
    inv_count: wp.float32,
    velocity_sum: wp.array[wp.vec3f],
    angular_velocity_sum: wp.array[wp.vec3f],
    kept_velocity: wp.array[wp.vec3f],
    kept_angular_velocity: wp.array[wp.vec3f],
):
    """Keep each body's final velocity and expose its substep average to position integration."""
    b = wp.tid()
    kept_velocity[b] = bodies.velocity[b]
    kept_angular_velocity[b] = bodies.angular_velocity[b]
    # Jointed bodies receive outer-step forces; their owned contacts are solved
    # after the temporal sweeps, so intermediate velocities lack that reaction.
    if jointed[b] == 0:
        bodies.velocity[b] = velocity_sum[b] * inv_count
        bodies.angular_velocity[b] = angular_velocity_sum[b] * inv_count


@wp.kernel(enable_backward=False)
def temporal_restore_velocity_kernel(
    bodies: BodyContainer,
    kept_velocity: wp.array[wp.vec3f],
    kept_angular_velocity: wp.array[wp.vec3f],
):
    """Restore every body's final sweep velocity after position integration."""
    b = wp.tid()
    bodies.velocity[b] = kept_velocity[b]
    bodies.angular_velocity[b] = kept_angular_velocity[b]


@wp.kernel(enable_backward=False)
def temporal_initialize_weld_components_kernel(entries: wp.array[wp.int64], marked: wp.array[wp.int32]):
    """Start one unmarked component per PhoenX body slot."""
    b = wp.tid()
    entries[b] = _pack_entry(b, wp.int32(0))
    marked[b] = 0


@wp.kernel(enable_backward=False)
def temporal_union_weld_components_kernel(
    enabled: wp.array[wp.bool],
    kinds: wp.array[wp.int32],
    parents: wp.array[wp.int32],
    children: wp.array[wp.int32],
    dims: wp.array2d[wp.int32],
    free_kind: int,
    entries: wp.array[wp.int64],
):
    """Join bodies connected by enabled joints with no free axes."""
    j = wp.tid()
    if not enabled[j] or kinds[j] == free_kind:
        return
    if dims[j, 0] + dims[j, 1] == 0 and parents[j] >= 0 and children[j] >= 0:
        _unite(entries, parents[j] + 1, children[j] + 1)


@wp.kernel(enable_backward=False)
def temporal_mark_jointed_components_kernel(
    enabled: wp.array[wp.bool],
    kinds: wp.array[wp.int32],
    parents: wp.array[wp.int32],
    children: wp.array[wp.int32],
    dims: wp.array2d[wp.int32],
    free_kind: int,
    entries: wp.array[wp.int64],
    marked: wp.array[wp.int32],
):
    """Mark entire weld components that touch an enabled free-axis joint."""
    j = wp.tid()
    if not enabled[j] or kinds[j] == free_kind or children[j] < 0:
        return
    if dims[j, 0] + dims[j, 1] > 0:
        wp.atomic_max(marked, _find(entries, children[j] + 1), 1)
        if parents[j] >= 0:
            wp.atomic_max(marked, _find(entries, parents[j] + 1), 1)


@wp.kernel(enable_backward=False)
def temporal_scatter_jointed_kernel(
    entries: wp.array[wp.int64], marked: wp.array[wp.int32], result: wp.array[wp.int32]
):
    """Apply each completed component's force schedule to all its bodies."""
    b = wp.tid()
    result[b] = marked[_find(entries, b)]


@wp.kernel(enable_backward=False)
def temporal_apply_forces_kernel(
    bodies: BodyContainer,
    gravity: wp.array[wp.vec3f],
    jointed: wp.array[wp.int32],
    free_dt: wp.float32,
    jointed_dt: wp.float32,
):
    """Apply external forces and gravity over ``free_dt`` or, for ``jointed`` bodies, ``jointed_dt``.

    Mirrors the force part of the substep-entry kernel for awake dynamic bodies.
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_DYNAMIC or bodies.inverse_mass[i] == 0.0 or bodies.island_root[i] >= 0:
        return
    dt = free_dt
    if jointed[i] != wp.int32(0):
        dt = jointed_dt
    if dt == wp.float32(0.0):
        return
    v = bodies.velocity[i] + bodies.force[i] * (bodies.inverse_mass[i] * dt)
    w = bodies.angular_velocity[i] + (mat33_from_sym6(bodies.inverse_inertia_world[i]) * bodies.torque[i]) * dt
    if bodies.affected_by_gravity[i] != 0:
        v = v + gravity[bodies.world_id[i]] * dt
    bodies.velocity[i] = v
    bodies.angular_velocity[i] = w


__all__ = [
    "temporal_accumulate_velocity_kernel",
    "temporal_apply_forces_kernel",
    "temporal_contact_gap_kernel",
    "temporal_initialize_weld_components_kernel",
    "temporal_mark_jointed_components_kernel",
    "temporal_restore_velocity_kernel",
    "temporal_scatter_jointed_kernel",
    "temporal_swap_average_velocity_kernel",
    "temporal_union_weld_components_kernel",
]
