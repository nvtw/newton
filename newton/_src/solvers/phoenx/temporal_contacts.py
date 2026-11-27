# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Temporal contact iterations: each biased PGS sweep of a substep is a contact substep.

Smaller contact steps can reduce oscillation when heavy bodies rest on a
light movable support, such as bottles in a carton. A full solver
substep reruns joint factorization, direct-contact responses, and row
preparation. Temporal iterations keep that work once per substep and advance
contact rows and eligible authored local scalar drives between sweeps:

* each sweep applies ``1/n`` of the substep's gravity and external forces
  (bodies on joints with free axes receive them at the substep start, see
  :func:`temporal_mark_jointed_components_kernel`),
* contact rows use the soft spring coefficients of ``dt/n`` and are warm
  started with the previous sweep's impulses, which are impulses over ``dt/n``,
* each row's gap advances by its relative normal motion over ``dt/n`` and its
  bias follows the gap, as when a substep reuses prepared rows,
* positions integrate the average of the sweeps' velocities, which is the
  motion the contact gaps tracked; bodies keep the last sweep's velocity,
* contact columns that press on a light body shared by many rows take half of
  each sweep update (:func:`temporal_sweep_relaxation_kernel`).

Local scalar-drive components use the same microstep for gravity, spring damping,
and bounded motor impulses. Components touching exact mechanisms, fixed companions,
or unsupported joint features retain their outer-step drive schedule. Joint rows
and exact joint projections run after every sweep as before. With
``n = 1`` nothing changes.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.articulations.direct_equality import _apply_direct_equality_body
from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_pd_coefficients,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_solve import contact_normal_bias
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
from newton._src.solvers.phoenx.constraints.contact_restitution import accumulate_impact_delay, impact_sweep
from newton._src.solvers.phoenx.constraints.contact_warm_start import (
    apply_gathered_contact_impulse,
    gathered_contact_impulse,
    store_row_warm_start,
)
from newton._src.solvers.phoenx.islands.island_builder import _find, _pack_entry, _unite

# One CUDA warp advances one column, reducing bounds across four adjacent rows.
TEMPORAL_SURFACE_GAP_LANES = 32


@wp.func
def _advance_temporal_gap(
    cc: ContactContainer,
    k: int,
    v1: wp.vec3f,
    w1: wp.vec3f,
    v2: wp.vec3f,
    w2: wp.vec3f,
    advance_dt: float,
    idt: float,
    bias_rate: float,
):
    gap = cc.gap[k]
    if advance_dt > 0.0:
        relative = v2 + wp.cross(w2, cc_get_r1(cc, k)) - v1 - wp.cross(w1, cc_get_r0(cc, k))
        gap = gap + advance_dt * wp.dot(relative, cc_get_normal(cc, k))
        cc.gap[k] = gap
    bias = contact_normal_bias(gap, idt, bias_rate, cc.length_scale)
    cc_set_bias(cc, k, bias)
    return gap, bias


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    unsigned mask = __activemask();
    for (int offset = 1; offset < 4; offset *= 2) {
        bias = fminf(bias, __shfl_xor_sync(mask, bias, offset, 4));
        r0 = fmaxf(r0, __shfl_xor_sync(mask, r0, offset, 4));
        r1 = fmaxf(r1, __shfl_xor_sync(mask, r1, offset, 4));
    }
#endif
    return wp::vec_t<4, float>(bias, r0, r1, 0.0f);
""")
def _reduce_separation_bounds(bias: float, r0: float, r1: float) -> wp.vec4f: ...


@wp.kernel(enable_backward=False)
def temporal_surface_gap_kernel(
    columns: ContactColumnContainer,
    cc: ContactContainer,
    bodies: BodyContainer,
    column_count: wp.array[int],
    owner: wp.array[int],
    advance_dt: float,
    idt: float,
    bias_rate: float,
    column_stride: int,
):
    """Update moving-surface gaps and bound the approach speed of four unloaded rows at a time."""
    first_column, lane = wp.tid()
    for cid in range(first_column, column_count[0], column_stride):
        if owner[cid] >= 0:
            continue
        b1 = contact_get_body1(columns, cid)
        b2 = contact_get_body2(columns, cid)
        v1 = bodies.velocity[b1]
        w1 = bodies.angular_velocity[b1]
        v2 = bodies.velocity[b2]
        w2 = bodies.angular_velocity[b2]
        first = contact_get_contact_first(columns, cid)
        end = first + contact_get_contact_count(columns, cid)
        # Padding the last group contributes neutral values without reading the next column.
        for start in range(first, end, wp.static(TEMPORAL_SURFACE_GAP_LANES)):
            k = start + lane
            lower = wp.float32(1.0e25)
            r0 = wp.float32(0.0)
            r1 = wp.float32(0.0)
            if k < end:
                gap, bias = _advance_temporal_gap(cc, k, v1, w1, v2, w2, advance_dt, idt, bias_rate)
                lower = wp.float32(-1.0)
                if gap > 0.0 and cc.impulses[0, k] == 0.0 and cc.impulses[1, k] == 0.0 and cc.impulses[2, k] == 0.0:
                    if wp.length_sq(cc_get_normal(cc, k)) < wp.float32(1.00001):
                        target = cc.surface_target[k][2]
                        # Allow for normal normalization, dot-product roundoff and cancellation
                        # between the surface target and the speculative bias.
                        lower = (bias + target) * 0.99998 - 1.0e-4 - 2.0e-6 * (wp.abs(bias) + wp.abs(target))
                        r0 = wp.max(wp.length(cc_get_r0(cc, k)), wp.float32(1.0e-18))
                        r1 = wp.max(wp.length(cc_get_r1(cc, k)), wp.float32(1.0e-18))
            bounds = _reduce_separation_bounds(lower, r0, r1)
            if (lane & 3) == 0 and k < end:
                cc.separation_bounds[k] = bounds


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
    views: ContactViews,
    correction: wp.array2d[wp.int64],
    native_dt: float,
    sweep: int,
    sweeps: int,
    warm_start_linear: wp.array[wp.vec3f],
    warm_start_angular0: wp.array[wp.vec3f],
    warm_start_angular1: wp.array[wp.vec3f],
):
    """Advance each ordinary contact row's gap by ``advance_dt`` of relative normal motion; refresh its bias.

    ``idt`` and ``bias_rate`` belong to the contact substep: a separated row admits
    the approach that closes its gap within one contact substep. Authored material
    coefficients use that same timestep and the updated penetration depth. Non-empty
    ``warm_start_*`` arrays also receive the rows' impulses for the next sweep's
    warm start (see :func:`store_row_warm_start`), which the gap update leaves unchanged.
    """
    previous_start = native_dt * float(wp.max(0, sweep - 1)) / float(sweeps)
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
        refresh = warm_start_linear.shape[0] > 0 and columns.static_owner[cid] < wp.int32(0)
        for k in range(first + lane, first + count, rows_per_column):
            if refresh:
                store_row_warm_start(cc, k, warm_start_linear, warm_start_angular0, warm_start_angular1)
            gap, _bias = _advance_temporal_gap(cc, k, v1, w1, v2, w2, advance_dt, idt, bias_rate)

            # Contact views and uncolored rows share canonical ordering, also
            # when compound-body grouping sorts the input material arrays.
            if views.restitution_target.shape[0] > k:
                target = views.restitution_target[k]
                arrival = target[1]
                if target[0] > 0.0 or arrival != 0.0:
                    event = impact_sweep(arrival, native_dt, sweeps)
                    if advance_dt > 0.0 and arrival != 0.0 and event == sweep - 1:
                        accumulate_impact_delay(views, bodies, cc, correction, k, arrival - previous_start)
                    if event == sweep:
                        cc_set_bias(cc, k, 0.0)
                        cc_set_pd_gamma(cc, k, 0.0)
                        cc_set_pd_bias(cc, k, target[0])
                        cc_set_pd_eff_soft(cc, k, cc_get_eff_n(cc, k))
                    else:
                        cc_set_pd_eff_soft(cc, k, 0.0)
                    continue
                if target[0] < 0.0:
                    cc_set_pd_eff_soft(cc, k, 0.0)
            if (
                views.rigid_contact_stiffness.shape[0] > k
                or views.rigid_contact_damping.shape[0] > k
                or views.shape_compliance.shape[0] > 0
            ):
                eff_n = cc_get_eff_n(cc, k)
                gamma, bias, eff_soft = wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
                if gap <= 0.0 and eff_n > 0.0:
                    gamma, bias, eff_soft = contact_pd_coefficients(
                        views, k, -gap, wp.float32(1.0) / eff_n, wp.float32(1.0) / idt, cc
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
    owned_body_mechanism: wp.array[wp.int32],
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
    # Only owned direct contacts react after the temporal sweeps. Ordinary
    # contacts on jointed bodies already contribute to each sweep's motion.
    use_final = False
    if jointed[b] != 0 and b < owned_body_mechanism.shape[0]:
        use_final = owned_body_mechanism[b] >= 0
    if not use_final:
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
    bodies: BodyContainer,
    include_free_axes: wp.bool,
):
    """Join bodies connected by enabled joints with no free axes."""
    j = wp.tid()
    if not enabled[j] or kinds[j] == free_kind:
        return
    if (include_free_axes or dims[j, 0] + dims[j, 1] == 0) and parents[j] >= 0 and children[j] >= 0:
        if include_free_axes and (
            bodies.motion_type[parents[j] + 1] != MOTION_DYNAMIC
            or bodies.motion_type[children[j] + 1] != MOTION_DYNAMIC
        ):
            return
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
    outer: wp.array[wp.bool],
):
    """Mark entire weld components that touch an enabled free-axis joint."""
    j = wp.tid()
    if not enabled[j] or kinds[j] == free_kind or children[j] < 0:
        return
    mark = dims[j, 0] + dims[j, 1] > 0
    if outer.shape[0] > 0:
        mark = outer[j]
    if mark:
        wp.atomic_max(marked, _find(entries, children[j] + 1), 1)
        if parents[j] >= 0:
            wp.atomic_max(marked, _find(entries, parents[j] + 1), 1)


@wp.kernel(enable_backward=False)
def temporal_mark_free_axis_bodies_kernel(
    enabled: wp.array[wp.bool],
    kinds: wp.array[wp.int32],
    parents: wp.array[wp.int32],
    children: wp.array[wp.int32],
    dims: wp.array2d[wp.int32],
    free_kind: int,
    marked: wp.array[wp.int32],
):
    """Mark the PhoenX body slots of both bodies of every enabled free-axis joint."""
    j = wp.tid()
    if not enabled[j] or kinds[j] == free_kind or dims[j, 0] + dims[j, 1] == 0:
        return
    if children[j] >= 0:
        marked[children[j] + 1] = 1
    if parents[j] >= 0:
        marked[parents[j] + 1] = 1


@wp.kernel(enable_backward=False)
def temporal_scatter_jointed_kernel(
    entries: wp.array[wp.int64], marked: wp.array[wp.int32], result: wp.array[wp.int32]
):
    """Apply each completed component's force schedule to all its bodies."""
    b = wp.tid()
    result[b] = marked[_find(entries, b)]


@wp.func
def _temporal_apply_body_forces(
    bodies: BodyContainer,
    gravity: wp.array[wp.vec3f],
    jointed: wp.array[wp.int32],
    free_dt: wp.float32,
    jointed_dt: wp.float32,
    i: wp.int32,
):
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
    _temporal_apply_body_forces(bodies, gravity, jointed, free_dt, jointed_dt, wp.tid())


@wp.kernel(enable_backward=False)
def temporal_sweep_entry_kernel(
    bodies: BodyContainer,
    num_bodies: wp.int32,
    reset: wp.int32,
    velocity_sum: wp.array[wp.vec3f],
    angular_velocity_sum: wp.array[wp.vec3f],
    apply_forces: wp.int32,
    gravity: wp.array[wp.vec3f],
    jointed: wp.array[wp.int32],
    free_dt: wp.float32,
    jointed_dt: wp.float32,
    columns: ContactColumnContainer,
    body_entry_start: wp.array[wp.int32],
    entries: wp.array[wp.int32],
    row_linear: wp.array[wp.vec3f],
    row_angular0: wp.array[wp.vec3f],
    row_angular1: wp.array[wp.vec3f],
    local_body_count: wp.int32,
    local_body_row_start: wp.array[wp.int32],
    local_descriptor: wp.array[wp.vec2i],
    local_wrench0: wp.array[wp.spatial_vector],
    local_wrench1: wp.array[wp.spatial_vector],
    local_seed: wp.array[wp.float32],
    local_scale: wp.array[wp.float32],
):
    """Close one contact substep and enter the next for one body, eight lanes per body.

    Performs, in order, :func:`temporal_accumulate_velocity_kernel`,
    :func:`temporal_apply_forces_kernel` (when ``apply_forces``), the
    cached contact warm start of :class:`ContactBodyEntries`, and the local
    joint microstep warm start for bodies below ``local_body_count``. Each
    step touches only its own body, so the fused result matches the
    separate launches.
    """
    tid = wp.tid()
    body = tid >> wp.int32(3)
    lane = tid & wp.int32(7)
    inverse_mass = bodies.inverse_mass[body]
    linear = wp.vec3f(0.0)
    angular = wp.vec3f(0.0)
    if inverse_mass > wp.float32(0.0):
        linear, angular = gathered_contact_impulse(
            tid, columns, body_entry_start, entries, row_linear, row_angular0, row_angular1
        )
    if lane != wp.int32(0):
        return
    if body < num_bodies:
        if reset != wp.int32(0):
            velocity_sum[body] = bodies.velocity[body]
            angular_velocity_sum[body] = bodies.angular_velocity[body]
        else:
            velocity_sum[body] = velocity_sum[body] + bodies.velocity[body]
            angular_velocity_sum[body] = angular_velocity_sum[body] + bodies.angular_velocity[body]
        if apply_forces != wp.int32(0):
            _temporal_apply_body_forces(bodies, gravity, jointed, free_dt, jointed_dt, body)
    if inverse_mass > wp.float32(0.0):
        apply_gathered_contact_impulse(bodies, body, inverse_mass, linear, angular)
    if body < local_body_count and bodies.island_root[body] < 0:
        _apply_direct_equality_body(
            body,
            local_body_row_start,
            local_descriptor,
            local_wrench0,
            local_wrench1,
            local_seed,
            local_scale,
            bodies,
        )


__all__ = [
    "temporal_accumulate_velocity_kernel",
    "temporal_apply_forces_kernel",
    "temporal_contact_gap_kernel",
    "temporal_count_body_rows_kernel",
    "temporal_initialize_weld_components_kernel",
    "temporal_mark_free_axis_bodies_kernel",
    "temporal_mark_jointed_components_kernel",
    "temporal_restore_velocity_kernel",
    "temporal_scatter_jointed_kernel",
    "temporal_surface_gap_kernel",
    "temporal_swap_average_velocity_kernel",
    "temporal_sweep_entry_kernel",
    "temporal_sweep_relaxation_kernel",
    "temporal_union_weld_components_kernel",
]


@wp.kernel(enable_backward=False)
def temporal_select_local_joints(
    enabled: wp.array[wp.bool],
    local: wp.array[wp.bool],
    parent: wp.array[wp.int32],
    child: wp.array[wp.int32],
    entries: wp.array[wp.int64],
    marked: wp.array[wp.int32],
    selected: wp.array[wp.bool],
    bodies: BodyContainer,
):
    joint = wp.tid()
    use_micro = enabled[joint] and local[joint]
    awake = wp.bool(False)
    for side in range(2):
        body = wp.where(side == 0, parent[joint], child[joint]) + 1
        if body > 0 and bodies.motion_type[body] == MOTION_DYNAMIC:
            if bodies.island_root[body] < 0:
                awake = True
            if marked[_find(entries, body)] != 0:
                use_micro = False
    selected[joint] = use_micro and awake


@wp.kernel(enable_backward=False)
def temporal_clear_local_force_flags(
    entries: wp.array[wp.int64],
    marked: wp.array[wp.int32],
    jointed: wp.array[wp.int32],
):
    body = wp.tid()
    if marked[_find(entries, body)] == 0:
        jointed[body] = 0


@wp.kernel(enable_backward=False)
def temporal_union_contact_components(
    bodies: BodyContainer,
    cols: ContactColumnContainer,
    ids: wp.array[wp.int32],
    active: wp.array[wp.int32],
    offset: int,
    entries: wp.array[wp.int64],
):
    slot = wp.tid()
    if slot >= active[0]:
        return
    cid = ids[slot] - offset
    if cid < 0:
        return
    first, second = contact_get_body1(cols, cid), contact_get_body2(cols, cid)
    if (
        first > 0
        and second > 0
        and bodies.motion_type[first] == MOTION_DYNAMIC
        and bodies.motion_type[second] == MOTION_DYNAMIC
    ):
        _unite(entries, first, second)


@wp.kernel(enable_backward=False)
def temporal_count_body_rows_kernel(
    columns: ContactColumnContainer,
    cc: ContactContainer,
    bodies: BodyContainer,
    column_count: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    body_rows: wp.array[wp.int32],
):
    """Count every dynamic body's penetrating ordinary contact rows."""
    cid = wp.tid()
    if cid >= column_count[0] or owner[cid] >= wp.int32(0):
        return
    b1 = contact_get_body1(columns, cid)
    b2 = contact_get_body2(columns, cid)
    first = contact_get_contact_first(columns, cid)
    count = contact_get_contact_count(columns, cid)
    rows = wp.int32(0)
    for k in range(first, first + count):
        if cc.gap[k] <= wp.float32(0.0):
            rows += wp.int32(1)
    if rows > wp.int32(0):
        if bodies.inverse_mass[b1] > wp.float32(0.0):
            wp.atomic_add(body_rows, b1, rows)
        if bodies.inverse_mass[b2] > wp.float32(0.0):
            wp.atomic_add(body_rows, b2, rows)


@wp.kernel(enable_backward=False)
def temporal_sweep_relaxation_kernel(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    column_count: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    body_rows: wp.array[wp.int32],
    joint_body: wp.array[wp.int32],
    budget: wp.float32,
    relaxation: wp.array[wp.float32],
):
    """Halve the sweep update of rows that load a light body shared by many contact rows.

    A biased sweep of a contact substep corrects the fraction ``m`` (the soft
    mass coefficient) of each row's velocity error, assuming the row's bodies
    respond to it alone. Rows that press on one body all correct it within the
    same Gauss-Seidel sweep. Weighted by each body's share of the row's inverse
    mass, a column's coupling ``d`` counts the penetrating rows that share its
    bodies; ``d`` is large for heavy contents on a light support. While
    ``m * d`` exceeds ``budget`` the single sweep overshoots that support and
    feeds a mode that turns by about 0.9 rad per contact substep, so such
    columns take half of each update (Gauss-Seidel under-relaxation with
    weight 1/2). The weight scales the whole update, so a converged sweep, and
    with it the contact spring, is unchanged; weakly coupled rows keep the
    exact soft step. Both bodies receive the same scaled impulse, so momentum
    is conserved.

    The coupling uses each body's free inverse mass, which describes a free body
    and, conservatively, a part of a welded composite. A body on a free-axis
    joint or in a weld component with one (``joint_body``, e.g. a driven
    actuator) moves as the joint and drive solves between the sweeps dictate;
    the estimate does not describe it, and its columns keep the full update.
    """
    cid = wp.tid()
    if cid >= column_count[0]:
        return
    first = contact_get_contact_first(columns, cid)
    count = contact_get_contact_count(columns, cid)
    w = wp.float32(1.0)
    b1 = contact_get_body1(columns, cid)
    b2 = contact_get_body2(columns, cid)
    if owner[cid] < wp.int32(0) and joint_body[b1] == 0 and joint_body[b2] == 0:
        a1 = bodies.inverse_mass[b1]
        a2 = bodies.inverse_mass[b2]
        if a1 + a2 > wp.float32(0.0):
            coupling = (wp.float32(body_rows[b1]) * a1 + wp.float32(body_rows[b2]) * a2) / (a1 + a2)
            if coupling > budget:
                w = wp.float32(0.5)
    if count > 0:
        relaxation[first] = w
