# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels for :class:`PhoenXWorld`.

Split from :mod:`solver_phoenx` so the driver class stays readable.
Dispatches only the two constraint types the solver supports:
:data:`CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET` and
:data:`CONSTRAINT_TYPE_CONTACT`.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import (
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
)
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    actuated_double_ball_socket_iterate,
    actuated_double_ball_socket_prepare_for_iteration,
    actuated_double_ball_socket_world_error,
    actuated_double_ball_socket_world_wrench,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactViews,
    contact_iterate,
    contact_position_iterate,
    contact_prepare_for_iteration,
    contact_world_error,
    contact_world_wrench,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET,
    CONSTRAINT_TYPE_CONTACT,
    ConstraintContainer,
    constraint_get_body1,
    constraint_get_body2,
    constraint_get_type,
)
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    ElementInteractionData,
    element_interaction_data_make,
)

__all__ = [
    "_STRAGGLER_BLOCK_DIM",
    "_constraint_gather_errors_kernel",
    "_constraint_gather_wrenches_kernel",
    "_constraint_iterate_fast_tail_kernel",
    "_constraint_position_iterate_fast_tail_kernel",
    "_constraint_prepare_fast_tail_kernel",
    "_constraint_relax_fast_tail_kernel",
    "_constraints_to_elements_kernel",
    "_integrate_velocities_kernel",
    "_kinematic_interpolate_substep_kernel",
    "_kinematic_prepare_step_kernel",
    "_rotation_quaternion",
    "_set_kinematic_pose_batch_kernel",
    "_world_csr_count_kernel",
    "_world_csr_prefix_offsets_kernel",
    "_world_csr_scan_kernel",
    "_world_csr_scatter_kernel",
    "pack_body_xforms_kernel",
]


_STRAGGLER_BLOCK_DIM: int = 256


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


# ---------------------------------------------------------------------------
# Per-world CSR bucketing
# ---------------------------------------------------------------------------
#
# The incremental partitioner produces a single global Jones-Plassmann
# coloring over every active constraint. Worlds share no bodies, so
# per-world slices can be carved out in a cheap post-pass instead of
# colouring once per world.
#
# Pipeline: ``_world_csr_count_kernel`` -> ``_world_csr_scan_kernel``
# -> ``_world_csr_prefix_offsets_kernel`` -> ``_world_csr_scatter_kernel``.
# Output layout: ``world_element_ids_by_color[capacity]`` (flat,
# bucketed by world); ``world_color_starts[w, c]`` (per-world
# exclusive-prefix); ``world_csr_offsets[w]`` (base index for world
# ``w``'s slice); ``world_num_colors[w]`` (highest non-empty colour).


@wp.func
def _find_color_of_position(
    color_starts: wp.array[wp.int32],
    num_colors: wp.int32,
    pos: wp.int32,
) -> wp.int32:
    """Return ``c`` s.t. ``color_starts[c] <= pos < color_starts[c + 1]``.
    ``O(log num_colors)`` binary search."""
    lo = wp.int32(0)
    hi = num_colors
    while lo < hi:
        mid = (lo + hi) >> 1
        if color_starts[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


@wp.kernel(enable_backward=False)
def _world_csr_count_kernel(
    bodies: BodyContainer,
    elements: wp.array[ElementInteractionData],
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors_arr: wp.array[wp.int32],
    # out
    world_color_counts: wp.array2d[wp.int32],
):
    """Atomic count of per-(world, colour) elements.

    One thread per position in the global ``element_ids_by_color``.
    Reads the compacted primary body from ``elements[cid].bodies[0]``
    so worlds sharing a static world body still route via the dynamic
    body's world id.
    """
    tid = wp.tid()
    n_colors = num_colors_arr[0]
    if n_colors == 0:
        return
    total_elements = color_starts[n_colors]
    if tid >= total_elements:
        return
    c = _find_color_of_position(color_starts, n_colors, tid)
    cid = element_ids_by_color[tid]
    b_primary = elements[cid].bodies[0]
    if b_primary < 0:
        return
    w = bodies.world_id[b_primary]
    wp.atomic_add(world_color_counts, w, c, wp.int32(1))


@wp.kernel(enable_backward=False)
def _world_csr_scan_kernel(
    world_color_counts: wp.array2d[wp.int32],
    num_colors_arr: wp.array[wp.int32],
    # out
    world_color_starts: wp.array2d[wp.int32],
    world_num_colors: wp.array[wp.int32],
):
    """Per-world serial prefix scan -> ``world_color_starts`` +
    ``world_num_colors`` (highest non-empty colour index). One thread
    per world."""
    w = wp.tid()
    n_colors = num_colors_arr[0]
    acc = wp.int32(0)
    highest = wp.int32(0)
    for c in range(n_colors):
        world_color_starts[w, c] = acc
        count = world_color_counts[w, c]
        if count > 0:
            highest = c + 1
        acc += count
    world_color_starts[w, n_colors] = acc
    world_num_colors[w] = highest


@wp.kernel(enable_backward=False)
def _world_csr_prefix_offsets_kernel(
    world_color_starts: wp.array2d[wp.int32],
    num_colors_arr: wp.array[wp.int32],
    num_worlds: wp.int32,
    # out
    world_csr_offsets: wp.array[wp.int32],
):
    """Single-thread prefix scan of per-world totals into
    ``world_csr_offsets`` (length ``num_worlds + 1``)."""
    tid = wp.tid()
    if tid != 0:
        return
    n_colors = num_colors_arr[0]
    acc = wp.int32(0)
    world_csr_offsets[0] = wp.int32(0)
    for w in range(num_worlds):
        acc += world_color_starts[w, n_colors]
        world_csr_offsets[w + 1] = acc


@wp.kernel(enable_backward=False)
def _world_csr_scatter_kernel(
    bodies: BodyContainer,
    elements: wp.array[ElementInteractionData],
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors_arr: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    # scratch (caller zeroes; atomic cursors -- one per (world, colour))
    world_color_cursor: wp.array2d[wp.int32],
    # out
    world_element_ids_by_color: wp.array[wp.int32],
):
    """Scatter each global-CSR element into its per-world slot.

    Runs after :func:`_world_csr_scan_kernel`. Atomic-cursor bump per
    ``(world, colour)`` gives a unique within-colour index; within-
    colour order is non-deterministic but the partitioner already
    guarantees no two same-colour elements share a body.
    """
    tid = wp.tid()
    n_colors = num_colors_arr[0]
    if n_colors == 0:
        return
    total_elements = color_starts[n_colors]
    if tid >= total_elements:
        return
    c = _find_color_of_position(color_starts, n_colors, tid)
    cid = element_ids_by_color[tid]
    b_primary = elements[cid].bodies[0]
    if b_primary < 0:
        return
    w = bodies.world_id[b_primary]
    local_slot = wp.atomic_add(world_color_cursor, w, c, wp.int32(1))
    dst = world_csr_offsets[w] + world_color_starts[w, c] + local_slot
    world_element_ids_by_color[dst] = cid


# ---------------------------------------------------------------------------
# Fast-path unified single-block dispatchers
# ---------------------------------------------------------------------------
#
# One block per world; each block walks its world's full CSR internally.
# Inside each block: outer iteration loop, middle colour sweep with
# ``__syncthreads`` between colours, inner block-stride lane loop.
# Partitioner guarantees no two same-colour elements share a body, so
# per-lane RMW on body velocities is race-free.


@wp.kernel(enable_backward=False)
def _constraint_iterate_fast_tail_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
    num_iterations: wp.int32,
):
    """Main-solve dispatcher: ``num_iterations`` PGS sweeps per world,
    positional bias ON."""
    tid = wp.tid()
    local_tid = tid % _STRAGGLER_BLOCK_DIM
    world_id = tid / _STRAGGLER_BLOCK_DIM

    n_colors = world_num_colors[world_id]
    world_base = world_csr_offsets[world_id]

    it = wp.int32(0)
    while it < num_iterations:
        c = wp.int32(0)
        while c < n_colors:
            start = world_base + world_color_starts[world_id, c]
            end = world_base + world_color_starts[world_id, c + 1]
            count = end - start

            base = local_tid
            while base < count:
                cid = world_element_ids_by_color[start + base]
                t = constraint_get_type(constraints, cid)
                if t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
                    actuated_double_ball_socket_iterate(constraints, cid, bodies, idt, True)
                elif t == CONSTRAINT_TYPE_CONTACT:
                    contact_iterate(constraints, cid, bodies, idt, cc, contacts, True)
                base += _STRAGGLER_BLOCK_DIM

            _sync_threads()
            c += 1

        it += 1


@wp.kernel(enable_backward=False)
def _constraint_prepare_fast_tail_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
):
    """Prepare dispatcher: one sweep per world. Computes effective
    masses, velocity bias, applies warm-start impulse."""
    tid = wp.tid()
    local_tid = tid % _STRAGGLER_BLOCK_DIM
    world_id = tid / _STRAGGLER_BLOCK_DIM

    n_colors = world_num_colors[world_id]
    world_base = world_csr_offsets[world_id]

    c = wp.int32(0)
    while c < n_colors:
        start = world_base + world_color_starts[world_id, c]
        end = world_base + world_color_starts[world_id, c + 1]
        count = end - start

        base = local_tid
        while base < count:
            cid = world_element_ids_by_color[start + base]
            t = constraint_get_type(constraints, cid)
            if t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
                actuated_double_ball_socket_prepare_for_iteration(constraints, cid, bodies, idt)
            elif t == CONSTRAINT_TYPE_CONTACT:
                contact_prepare_for_iteration(constraints, cid, bodies, idt, cc, contacts)
            base += _STRAGGLER_BLOCK_DIM

        _sync_threads()
        c += 1


@wp.kernel(enable_backward=False)
def _constraint_position_iterate_fast_tail_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    world_element_ids_by_color: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    cc: ContactContainer,
    num_iterations: wp.int32,
):
    """XPBD position-iteration dispatcher for contact tangent drift.

    Runs between ``integrate_velocities`` and ``relax_velocities``.
    Each per-slot position iterate is gated by the slip threshold so
    sliding pairs are skipped; the Coulomb-clamped velocity friction
    handles them.
    """
    tid = wp.tid()
    local_tid = tid % _STRAGGLER_BLOCK_DIM
    world_id = tid / _STRAGGLER_BLOCK_DIM

    n_colors = world_num_colors[world_id]
    world_base = world_csr_offsets[world_id]

    it = wp.int32(0)
    while it < num_iterations:
        c = wp.int32(0)
        while c < n_colors:
            start = world_base + world_color_starts[world_id, c]
            end = world_base + world_color_starts[world_id, c + 1]
            count = end - start

            base = local_tid
            while base < count:
                cid = world_element_ids_by_color[start + base]
                t = constraint_get_type(constraints, cid)
                if t == CONSTRAINT_TYPE_CONTACT:
                    contact_position_iterate(constraints, cid, bodies, cc)
                base += _STRAGGLER_BLOCK_DIM

            _sync_threads()
            c += 1

        it += 1


@wp.kernel(enable_backward=False)
def _constraint_relax_fast_tail_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
    num_iterations: wp.int32,
):
    """Box2D v3 TGS-soft relax dispatcher: ``num_iterations`` sweeps
    with positional bias OFF (enforces ``Jv = 0``)."""
    tid = wp.tid()
    local_tid = tid % _STRAGGLER_BLOCK_DIM
    world_id = tid / _STRAGGLER_BLOCK_DIM

    n_colors = world_num_colors[world_id]
    world_base = world_csr_offsets[world_id]

    it = wp.int32(0)
    while it < num_iterations:
        c = wp.int32(0)
        while c < n_colors:
            start = world_base + world_color_starts[world_id, c]
            end = world_base + world_color_starts[world_id, c + 1]
            count = end - start

            base = local_tid
            while base < count:
                cid = world_element_ids_by_color[start + base]
                t = constraint_get_type(constraints, cid)
                if t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
                    actuated_double_ball_socket_iterate(constraints, cid, bodies, idt, False)
                elif t == CONSTRAINT_TYPE_CONTACT:
                    contact_iterate(constraints, cid, bodies, idt, cc, contacts, False)
                base += _STRAGGLER_BLOCK_DIM

            _sync_threads()
            c += 1

        it += 1


@wp.kernel(enable_backward=False)
def _constraints_to_elements_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    num_constraints: wp.array[wp.int32],
    elements: wp.array[ElementInteractionData],
):
    """Project every active constraint into ``ElementInteractionData``.

    Only the two body indices matter to the graph colourer; static
    bodies get collapsed to ``-1`` and the dynamic body (if any) is
    compacted to slot 0.
    """
    tid = wp.tid()
    n = num_constraints[0]
    if tid >= n:
        elements[tid] = element_interaction_data_make(-1, -1, -1, -1, -1, -1, -1, -1)
        return
    b1 = constraint_get_body1(constraints, tid)
    b2 = constraint_get_body2(constraints, tid)
    if b1 >= 0 and bodies.inverse_mass[b1] == 0.0:
        b1 = -1
    if b2 >= 0 and bodies.inverse_mass[b2] == 0.0:
        b2 = -1
    # Compact: non-negative IDs must come first so the adjacency loop
    # (which stops on the first -1) doesn't miss a dynamic body when
    # the static one happens to sit in slot 0.
    if b1 < 0 and b2 >= 0:
        b1 = b2
        b2 = -1
    elements[tid] = element_interaction_data_make(b1, b2, -1, -1, -1, -1, -1, -1)


@wp.kernel(enable_backward=False)
def _constraint_gather_wrenches_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    num_constraints: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    out: wp.array[wp.spatial_vector],
):
    """Per-cid world-frame wrench on ``body2``: ``top = force [N]``,
    ``bottom = torque [N·m]``. ``idt = 1 / substep_dt``."""
    cid = wp.tid()
    if cid >= num_constraints:
        return
    t = constraint_get_type(constraints, cid)
    force = wp.vec3f(0.0, 0.0, 0.0)
    torque = wp.vec3f(0.0, 0.0, 0.0)
    if t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
        force, torque = actuated_double_ball_socket_world_wrench(constraints, cid, idt)
    elif t == CONSTRAINT_TYPE_CONTACT:
        force, torque = contact_world_wrench(constraints, cid, idt, cc, contacts)
    out[cid] = wp.spatial_vector(force, torque)


@wp.kernel(enable_backward=False)
def _constraint_gather_errors_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    num_constraints: wp.int32,
    # out
    out: wp.array[wp.spatial_vector],
):
    """Per-cid position-level constraint residual: ``top`` = linear
    [m], ``bottom`` = angular [rad]. Pure read from current body pose
    + persisted per-type state; no body mutation."""
    cid = wp.tid()
    if cid >= num_constraints:
        return
    t = constraint_get_type(constraints, cid)
    zero = wp.spatial_vector(wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.0, 0.0, 0.0))
    err = zero
    if t == CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET:
        err = actuated_double_ball_socket_world_error(constraints, cid, bodies)
    elif t == CONSTRAINT_TYPE_CONTACT:
        err = contact_world_error(constraints, cid)
    out[cid] = err


@wp.func
def _rotation_quaternion(omega: wp.vec3f, dt: wp.float32) -> wp.quatf:
    """Axis-angle rotation quaternion for ``omega * dt``. Unit norm by
    construction, stable across many substeps."""
    omega_len = wp.length(omega)
    theta = omega_len * dt
    if theta < 1.0e-9:
        return wp.quatf(0.0, 0.0, 0.0, 1.0)
    half = theta * 0.5
    s = wp.sin(half) / omega_len
    return wp.quatf(omega[0] * s, omega[1] * s, omega[2] * s, wp.cos(half))


@wp.kernel(enable_backward=False)
def _integrate_velocities_kernel(
    bodies: BodyContainer,
    dt: wp.float32,
):
    """Advance position + orientation for dynamic bodies only.

    Static bodies skipped unconditionally. Kinematic bodies are
    *also* skipped here -- their pose advances via explicit lerp /
    slerp interpolation between ``position_prev`` and
    ``kinematic_target_pos`` in
    :func:`_kinematic_interpolate_substep_kernel`, so running the
    velocity integration on them would double-advance the pose.
    """
    i = wp.tid()
    mt = bodies.motion_type[i]
    if mt == MOTION_STATIC or mt == MOTION_KINEMATIC:
        return

    bodies.position[i] = bodies.position[i] + bodies.velocity[i] * dt
    q_rot = _rotation_quaternion(bodies.angular_velocity[i], dt)
    bodies.orientation[i] = wp.normalize(q_rot * bodies.orientation[i])


@wp.kernel(enable_backward=False)
def _kinematic_prepare_step_kernel(
    bodies: BodyContainer,
    dt: wp.float32,
):
    """Once-per-step kinematic prepare, called at
    :meth:`PhoenXWorld.step` entry *before* the substep loop.

    Resolves this step's pose target for every kinematic body, infers
    the linear + angular velocity the solver should expose to contacts,
    and snapshots the body's current pose as ``position_prev`` /
    ``orientation_prev`` so the per-substep interpolator has a stable
    ``lerp`` / ``slerp`` origin.

    Target resolution:

    * ``kinematic_target_valid[i] == 1`` (user called
      :meth:`PhoenXWorld.set_kinematic_pose` or the Newton adapter
      flagged a pose import) -- read the user-set target out of
      ``kinematic_target_{pos,orient}`` and clear the flag.
    * ``kinematic_target_valid[i] == 0`` (no explicit script this
      step, constant-velocity backward-compat path) -- synthesise
      a target from ``position_prev + velocity * dt`` and the
      axis-angle integration of ``angular_velocity * dt``.

    Velocity inference uses the quaternion log-map so large
    rotations are handled correctly (small-angle ``omega ~= 2 *
    q_rel.xyz / dt`` is exact only at the limit; the full formula
    ``angle = 2 * atan2(|xyz|, w); omega = axis * angle / dt``
    generalises without drift).
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_KINEMATIC:
        return

    pos_prev = bodies.position[i]
    orient_prev = bodies.orientation[i]
    bodies.position_prev[i] = pos_prev
    bodies.orientation_prev[i] = orient_prev

    if bodies.kinematic_target_valid[i] == 1:
        pos_target = bodies.kinematic_target_pos[i]
        orient_target = bodies.kinematic_target_orient[i]
        # One-shot consumption: the user must re-assert the target
        # each step (this matches the Newton adapter, which flags
        # valid=1 every import).
        bodies.kinematic_target_valid[i] = 0
    else:
        # Constant-velocity path: advance from ``(pos, orient)`` by
        # ``(velocity, angular_velocity) * dt``. Uses the same
        # axis-angle rotation the dynamic integrator uses so a
        # constant-omega kinematic traces exactly the same orientation
        # trajectory as the legacy code.
        pos_target = pos_prev + bodies.velocity[i] * dt
        q_rot = _rotation_quaternion(bodies.angular_velocity[i], dt)
        orient_target = wp.normalize(q_rot * orient_prev)
        bodies.kinematic_target_pos[i] = pos_target
        bodies.kinematic_target_orient[i] = orient_target

    # Infer velocity from pose delta. For the constant-velocity path
    # this round-trips exactly (target = pos_prev + velocity * dt ->
    # inferred velocity == original velocity). For the scripted path
    # it exposes the pose-derivative for contact response.
    inv_dt = wp.float32(1.0) / dt
    v = (pos_target - pos_prev) * inv_dt

    # Angular velocity via log-map of ``q_rel = target * inv(prev)``.
    # Canonicalise to the shortest-path hemisphere first so the
    # ``atan2`` branch cut doesn't flip the sign.
    q_rel = orient_target * wp.quat_inverse(orient_prev)
    if q_rel[3] < 0.0:
        q_rel = -q_rel
    xyz = wp.vec3f(q_rel[0], q_rel[1], q_rel[2])
    xyz_len = wp.length(xyz)
    if xyz_len > 1.0e-9:
        angle = 2.0 * wp.atan2(xyz_len, q_rel[3])
        omega = xyz * (angle * inv_dt / xyz_len)
    else:
        omega = wp.vec3f(0.0, 0.0, 0.0)

    bodies.velocity[i] = v
    bodies.angular_velocity[i] = omega


@wp.kernel(enable_backward=False)
def _set_kinematic_pose_batch_kernel(
    bodies: BodyContainer,
    body_ids: wp.array[wp.int32],
    target_positions: wp.array[wp.vec3f],
    target_orientations: wp.array[wp.quatf],
):
    """Batched writeback for :meth:`PhoenXWorld.set_kinematic_pose`.

    One thread per entry in ``body_ids``. Writes the target pose into
    the kinematic-scripting slots and flags
    ``kinematic_target_valid = 1`` so the next
    :func:`_kinematic_prepare_step_kernel` picks it up.

    Attempting to script a non-kinematic body is a no-op (silent);
    callers should validate on the host side and raise clearly.
    """
    k = wp.tid()
    b = body_ids[k]
    if bodies.motion_type[b] != MOTION_KINEMATIC:
        return
    bodies.kinematic_target_pos[b] = target_positions[k]
    bodies.kinematic_target_orient[b] = target_orientations[k]
    bodies.kinematic_target_valid[b] = 1


@wp.kernel(enable_backward=False)
def _kinematic_interpolate_substep_kernel(
    bodies: BodyContainer,
    alpha: wp.float32,
):
    """Per-substep kinematic pose update.

    Called *after* :func:`_integrate_velocities_kernel` inside each
    substep with ``alpha = (substep_index + 1) / num_substeps``. Writes

    .. math::
        \\text{position}    &= \\text{lerp}(\\text{position}_{\\text{prev}},
                                 \\text{kinematic\\_target\\_pos}, \\alpha) \\\\
        \\text{orientation} &= \\text{slerp}(\\text{orientation}_{\\text{prev}},
                                 \\text{kinematic\\_target\\_orient}, \\alpha)

    At ``alpha = 1`` the body lands exactly on its target. Dynamic and
    static bodies are skipped (dynamic pose already advanced by
    :func:`_integrate_velocities_kernel`; static never moves).
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_KINEMATIC:
        return
    prev_pos = bodies.position_prev[i]
    target_pos = bodies.kinematic_target_pos[i]
    prev_orient = bodies.orientation_prev[i]
    target_orient = bodies.kinematic_target_orient[i]
    bodies.position[i] = (1.0 - alpha) * prev_pos + alpha * target_pos
    bodies.orientation[i] = wp.quat_slerp(prev_orient, target_orient, alpha)


@wp.kernel(enable_backward=False)
def pack_body_xforms_kernel(
    bodies: BodyContainer,
    xforms: wp.array[wp.transform],
):
    """Pack ``(position, orientation)`` into a flat ``wp.transform``
    array for ``viewer.log_shapes``."""
    i = wp.tid()
    xforms[i] = wp.transform(bodies.position[i], bodies.orientation[i])
