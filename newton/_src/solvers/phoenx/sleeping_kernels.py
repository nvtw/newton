# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sleeping-pipeline kernels for the PhoenX solver.

The pipeline keeps a persistent ``BodyContainer.island_root`` per body
(``-1`` = awake; any non-negative value is the lowest body id in the
island at the moment of sleep). Each step:

1. AABB fanin per body (for the angular score term).
2. Self-wake: sleeping bodies whose own predicted score is high (host-
   side velocity or force injection) flag their root for wake.
3. Apply wake flags.
4. Detect *active* sleeping islands -- any sleeping body sharing a
   constraint with an awake dynamic body marks its island as active.
5. Inject chain edges ``(b, root)`` for every body in an active
   island. These artificial edges pull the entire sleeping island back
   into the live union-find, so the awake body bridging it gets merged
   with every member. Inactive sleeping islands stay filtered out --
   they cost nothing in the live build.
6. Copy ``_elements`` to the 2D union-find buffer, dropping inactive
   sleeping bodies to ``-1`` and keeping active sleeping bodies (so the
   chain edges land correctly).
7. Union-find build (regular elements + chain edges).
8. Per-compact-island lowest body id, used to stamp new sleepers.
9. Per-island max velocity score; mark sleeping islands.
10. Propagate: a sleeping body in an awake compact island (i.e. the
    external bridge lifted the score) clears ``island_root`` -- whole
    island wakes atomically. An awake body whose island has been below
    threshold for ``sleeping_frames_required`` consecutive frames stamps
    ``island_root`` with the compact island's lowest body id.
11. Collapse ``_elements``: rewrite sleeping body slots to ``-1`` so
    the partitioner drops them from the solve.

External pre-collide wake (``PhoenXWorld.wake_on_external_input``) uses
the same fan + apply pattern but runs before ``model.collide`` so the
broad phase keeps the supporting contacts on the wake frame.

This module is intentionally separate from
:mod:`newton._src.solvers.phoenx.islands.island_builder` (the
``UnionFindIslandBuilder`` primitive) -- the builder stays generic;
all sleep-aware orchestration lives here.
"""

import warp as wp

from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
    mat33_from_sym6,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
)
from newton._src.solvers.phoenx.islands.island_builder import MAX_BODIES_PER_INTERACTION

__all__ = [
    "_phoenx_apply_island_wake_kernel",
    "_phoenx_apply_wake_flag_kernel",
    "_phoenx_collapse_sleeping_elements_kernel",
    "_phoenx_compute_island_root_per_compact_kernel",
    "_phoenx_copy_elements_to_int2d_kernel",
    "_phoenx_detect_active_islands_kernel",
    "_phoenx_finalize_body_aabb_diagonal_kernel",
    "_phoenx_init_body_aabb_kernel",
    "_phoenx_inject_chain_edges_kernel",
    "_phoenx_island_fanin_external_input_kernel",
    "_phoenx_island_max_velocity_kernel",
    "_phoenx_mark_sleeping_islands_kernel",
    "_phoenx_propagate_sleep_to_bodies_kernel",
    "_phoenx_seed_uf_num_interactions_kernel",
    "_phoenx_self_wake_fanin_kernel",
    "_phoenx_shape_aabb_fanin_kernel",
]


_AABB_INIT_LARGE = wp.constant(wp.float32(1.0e30))


# ---------------------------------------------------------------------------
# Per-body union AABB (for the angular term of the max-velocity score).
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _phoenx_init_body_aabb_kernel(
    body_aabb_lower: wp.array2d[wp.float32],
    body_aabb_upper: wp.array2d[wp.float32],
):
    """Per-body, per-axis reset to ±large so the fanin's atomic min/max
    converges to the union AABB of every attached shape."""
    b = wp.tid()
    body_aabb_lower[b, 0] = _AABB_INIT_LARGE
    body_aabb_lower[b, 1] = _AABB_INIT_LARGE
    body_aabb_lower[b, 2] = _AABB_INIT_LARGE
    body_aabb_upper[b, 0] = -_AABB_INIT_LARGE
    body_aabb_upper[b, 1] = -_AABB_INIT_LARGE
    body_aabb_upper[b, 2] = -_AABB_INIT_LARGE


@wp.kernel(enable_backward=False)
def _phoenx_shape_aabb_fanin_kernel(
    shape_aabb_lower: wp.array[wp.vec3f],
    shape_aabb_upper: wp.array[wp.vec3f],
    shape_body_phoenx: wp.array[wp.int32],
    body_aabb_lower: wp.array2d[wp.float32],
    body_aabb_upper: wp.array2d[wp.float32],
):
    """Union every attached shape's world-frame AABB into the body's AABB.
    ``shape_body_phoenx`` is PhoenX-shifted; slot 0 (anchor) is skipped."""
    s = wp.tid()
    nb = shape_body_phoenx[s]
    if nb <= 0:
        return
    lo = shape_aabb_lower[s]
    hi = shape_aabb_upper[s]
    wp.atomic_min(body_aabb_lower, nb, 0, lo[0])
    wp.atomic_min(body_aabb_lower, nb, 1, lo[1])
    wp.atomic_min(body_aabb_lower, nb, 2, lo[2])
    wp.atomic_max(body_aabb_upper, nb, 0, hi[0])
    wp.atomic_max(body_aabb_upper, nb, 1, hi[1])
    wp.atomic_max(body_aabb_upper, nb, 2, hi[2])


@wp.kernel(enable_backward=False)
def _phoenx_finalize_body_aabb_diagonal_kernel(
    body_aabb_lower: wp.array2d[wp.float32],
    body_aabb_upper: wp.array2d[wp.float32],
    body_aabb_diagonal: wp.array[wp.float32],
):
    """Per-body: diagonal = length(upper - lower); bodies with no shapes
    collapse the spin term to zero."""
    b = wp.tid()
    ux = body_aabb_upper[b, 0]
    uy = body_aabb_upper[b, 1]
    uz = body_aabb_upper[b, 2]
    lx = body_aabb_lower[b, 0]
    ly = body_aabb_lower[b, 1]
    lz = body_aabb_lower[b, 2]
    if ux < lx:
        body_aabb_diagonal[b] = wp.float32(0.0)
        return
    dx = ux - lx
    dy = uy - ly
    dz = uz - lz
    body_aabb_diagonal[b] = wp.sqrt(dx * dx + dy * dy + dz * dz)


# ---------------------------------------------------------------------------
# Active-island detection + chain-edge injection.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _phoenx_detect_active_islands_kernel(
    bodies: BodyContainer,
    elements: wp.array[ElementInteractionData],
    num_active_constraints: wp.array[wp.int32],
    num_bodies: wp.int32,
    # out: island_active[root] = 1 if some sleeping member of that island
    # shares a constraint with an awake dynamic body this step.
    island_active: wp.array[wp.int32],
):
    """A sleeping island is *active* this step iff at least one of its
    members shares a constraint with an awake source -- either an
    awake DYNAMIC body or a KINEMATIC body (user-driven mover, e.g. a
    camera collider). STATIC bodies (ground anchor) do NOT count as an
    awake source: they're always in contact with everything that
    settled on them, so treating them as awake would prevent any
    island from ever sleeping.
    """
    tid = wp.tid()
    if tid >= num_active_constraints[0]:
        return
    el = elements[tid]
    has_awake = wp.int32(0)
    has_sleeping = wp.int32(0)
    for j in range(MAX_BODIES):
        b = el.bodies[j]
        if b < 0:
            break
        if b >= num_bodies:
            continue
        mt = bodies.motion_type[b]
        if mt == MOTION_STATIC:
            continue
        if mt != MOTION_DYNAMIC:
            # KINEMATIC body (or any non-static, non-dynamic motion
            # type): always treated as an awake source.
            has_awake = wp.int32(1)
            continue
        if bodies.island_root[b] >= wp.int32(0):
            has_sleeping = wp.int32(1)
        else:
            has_awake = wp.int32(1)
    if has_awake == wp.int32(0) or has_sleeping == wp.int32(0):
        return
    for j in range(MAX_BODIES):
        b = el.bodies[j]
        if b < 0:
            break
        if b >= num_bodies:
            continue
        if bodies.motion_type[b] != MOTION_DYNAMIC:
            continue
        root = bodies.island_root[b]
        if root >= wp.int32(0):
            wp.atomic_max(island_active, root, wp.int32(1))


@wp.kernel(enable_backward=False)
def _phoenx_seed_uf_num_interactions_kernel(
    num_active_constraints: wp.array[wp.int32],
    uf_num_interactions: wp.array[wp.int32],
):
    """Copy ``num_active_constraints[0] -> uf_num_interactions[0]`` so the
    chain-edge injection can ``atomic_add`` past it. Single-threaded."""
    tid = wp.tid()
    if tid != 0:
        return
    uf_num_interactions[0] = num_active_constraints[0]


@wp.kernel(enable_backward=False)
def _phoenx_inject_chain_edges_kernel(
    bodies: BodyContainer,
    island_active: wp.array[wp.int32],
    interaction_capacity: wp.int32,
    # inout
    interaction_bodies: wp.array2d[wp.int32],
    uf_num_interactions: wp.array[wp.int32],
):
    """For each sleeping body in an active island, append an artificial
    edge ``(body, island_root)`` to the union-find input. The unite
    kernel then merges every active sleeping body with its root (and
    transitively with whichever awake body is bridging the island), so
    the live compact island ends up containing every sleeping member.

    Skips the root body's own slot (``b == root`` -> self-edge would be
    redundant). Inactive sleeping islands and awake bodies contribute
    nothing here.
    """
    b = wp.tid()
    if bodies.motion_type[b] != MOTION_DYNAMIC:
        return
    root = bodies.island_root[b]
    if root < wp.int32(0):
        return  # awake
    if root == b:
        return  # self-edge would be a no-op
    if island_active[root] == wp.int32(0):
        return  # island not touched this step; leave it ignored
    slot = wp.atomic_add(uf_num_interactions, 0, wp.int32(1))
    if slot >= interaction_capacity:
        # Capacity exhausted; rare. Drop the edge -- the body wakes
        # next frame via the contact path instead.
        return
    interaction_bodies[slot, 0] = b
    interaction_bodies[slot, 1] = root
    for j in range(2, MAX_BODIES_PER_INTERACTION):
        interaction_bodies[slot, j] = wp.int32(-1)


# ---------------------------------------------------------------------------
# Element -> 2D UF input.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _phoenx_copy_elements_to_int2d_kernel(
    elements: wp.array[ElementInteractionData],
    num_active_constraints: wp.array[wp.int32],
    num_bodies: wp.int32,
    bodies: BodyContainer,
    island_active: wp.array[wp.int32],
    interaction_bodies: wp.array2d[wp.int32],
):
    """Pack each element's dynamic + kinematic body slots to the front
    of its row in ``interaction_bodies``. Inactive cids, STATIC bodies,
    and particles are dropped to ``-1``.

    Sleeping (dynamic) bodies whose root is **not** active stay filtered
    (cost nothing in the live UF). Sleeping bodies in **active** islands
    are kept -- their chain edges to ``root`` (injected separately) then
    merge them with the bridging body. KINEMATIC bodies are kept so a
    kinematic mover (e.g. a camera collider) sharing an element with a
    sleeping brick lands in the same compact island; the per-island
    score kernel reads kinematic velocity and lifts the score above
    threshold, waking the island.
    """
    tid = wp.tid()
    if tid >= num_active_constraints[0]:
        for j in range(MAX_BODIES):
            interaction_bodies[tid, j] = wp.int32(-1)
        return
    el = elements[tid]
    write_idx = wp.int32(0)
    for j in range(MAX_BODIES):
        b = el.bodies[j]
        if b < 0 or b >= num_bodies:
            continue
        mt = bodies.motion_type[b]
        if mt != MOTION_DYNAMIC and mt != MOTION_KINEMATIC:
            continue
        if mt == MOTION_DYNAMIC:
            root = bodies.island_root[b]
            if root >= wp.int32(0) and island_active[root] == wp.int32(0):
                # Sleeping dynamic body whose island is inactive: skip.
                continue
        interaction_bodies[tid, write_idx] = b
        write_idx = write_idx + wp.int32(1)
    for j in range(write_idx, MAX_BODIES):
        interaction_bodies[tid, j] = wp.int32(-1)


# ---------------------------------------------------------------------------
# Per-island scoring + sleep marking.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _phoenx_island_max_velocity_kernel(
    bodies: BodyContainer,
    body_aabb_diagonal: wp.array[wp.float32],
    island_of_body: wp.array[wp.int32],
    step_dt: wp.float32,
    sleeping_velocity_threshold: wp.float32,
    island_max_velocity: wp.array[wp.float32],
):
    """Score = ``length(v_pred) + 0.5 * diag * length(w_pred)``, where
    ``v_pred`` folds one step of pending external wrench into velocity.
    Atomic-max into the body's compact island slot.

    Awake dynamic bodies contribute their own predicted velocity (force
    is folded in via ``v_pred = v + force * invm * dt``). Sleeping
    dynamic bodies are skipped -- their integration is paused so they
    always score 0; self-wake handles fresh force injection on them.

    KINEMATIC bodies contribute their inferred velocity (set by
    :func:`_kinematic_prepare_step_kernel` from pose-target delta or
    user-set body_qd). They have inv_mass==0 so force/torque can't
    accelerate them, but a moving kinematic mover (camera collider,
    scripted prop) must lift the island score so the compact island
    containing it wakes the sleeping bodies it's pressing against.
    """
    b = wp.tid()
    mt = bodies.motion_type[b]
    if mt != MOTION_DYNAMIC and mt != MOTION_KINEMATIC:
        return
    if bodies.island_root[b] >= wp.int32(0):
        return
    island = island_of_body[b]
    if island < 0:
        return
    if mt == MOTION_DYNAMIC:
        inv_mass = bodies.inverse_mass[b]
        if inv_mass == 0.0:
            return
        force = bodies.force[b]
        torque = bodies.torque[b]
        v_pred = bodies.velocity[b] + force * (inv_mass * step_dt)
        w_pred = bodies.angular_velocity[b] + (mat33_from_sym6(bodies.inverse_inertia_world[b]) * torque) * step_dt
    else:
        # KINEMATIC: inv_mass / inv_inertia are zero so external wrench
        # can't change velocity; use the inferred pose-derivative directly.
        force = wp.vec3f(0.0)
        torque = wp.vec3f(0.0)
        v_pred = bodies.velocity[b]
        w_pred = bodies.angular_velocity[b]
    score = wp.length(v_pred) + 0.5 * body_aabb_diagonal[b] * wp.length(w_pred)
    if score > island_max_velocity[island]:
        wp.atomic_max(island_max_velocity, island, score)


@wp.kernel(enable_backward=False)
def _phoenx_mark_sleeping_islands_kernel(
    island_max_velocity: wp.array[wp.float32],
    num_islands: wp.array[wp.int32],
    threshold: wp.float32,
    island_is_sleeping: wp.array[wp.int32],
):
    """``island_is_sleeping[i] = 1 iff island_max_velocity[i] < threshold``."""
    i = wp.tid()
    if i >= num_islands[0]:
        island_is_sleeping[i] = wp.int32(0)
        return
    if island_max_velocity[i] < threshold:
        island_is_sleeping[i] = wp.int32(1)
    else:
        island_is_sleeping[i] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _phoenx_compute_island_root_per_compact_kernel(
    bodies: BodyContainer,
    set_nr: wp.array[wp.int32],
    island_root_per_compact: wp.array[wp.int32],
):
    """``atomic_min(island_root_per_compact[set_nr[b]], b)`` for every
    awake dynamic body. Consumed by the propagate kernel to stamp a
    stable body-id label when a body's hysteresis saturates."""
    b = wp.tid()
    if bodies.motion_type[b] != MOTION_DYNAMIC:
        return
    if bodies.island_root[b] >= wp.int32(0):
        return
    nr = set_nr[b]
    if nr < 0:
        return
    wp.atomic_min(island_root_per_compact, nr, b)


@wp.kernel(enable_backward=False)
def _phoenx_propagate_sleep_to_bodies_kernel(
    bodies: BodyContainer,
    island_of_body: wp.array[wp.int32],
    island_is_sleeping: wp.array[wp.int32],
    island_root_per_compact: wp.array[wp.int32],
    sleeping_frames_required: wp.int32,
):
    """Three-way hysteresis dispatch:

    * Awake body, island awake: counter reset to 0.
    * Awake body, island below threshold: counter++ saturating; on
      saturation stamp ``island_root`` with
      ``island_root_per_compact[set_nr]`` (lowest awake body id in
      the compact island).
    * Sleeping body, island awake: clear ``island_root`` -- the
      external bridge + chain edges merged this island with a moving
      body, so every member wakes atomically.
    * Sleeping body, island below threshold: stay sleeping (counter
      pinned at the saturated value).
    """
    b = wp.tid()
    if bodies.motion_type[b] != MOTION_DYNAMIC:
        bodies.frames_below_threshold[b] = wp.int32(0)
        return
    if bodies.inverse_mass[b] == 0.0:
        bodies.frames_below_threshold[b] = wp.int32(0)
        return
    island = island_of_body[b]
    sleeping = bodies.island_root[b] >= wp.int32(0)
    if island < 0:
        # Body had no row in the union-find input. Only happens for
        # sleeping bodies in inactive islands (filtered out): leave
        # them sleeping. Awake bodies always get a singleton compact
        # id at minimum.
        if sleeping:
            bodies.frames_below_threshold[b] = sleeping_frames_required
        else:
            bodies.frames_below_threshold[b] = wp.int32(0)
        return
    island_awake = island_is_sleeping[island] == wp.int32(0)
    if island_awake:
        # The compact island contains a moving body. Wake everyone in
        # it atomically.
        bodies.frames_below_threshold[b] = wp.int32(0)
        if sleeping:
            bodies.island_root[b] = wp.int32(-1)
        return
    if sleeping:
        # Already sleeping, island still quiet. Stay sleeping.
        bodies.frames_below_threshold[b] = sleeping_frames_required
        return
    # Awake body, compact island below threshold. Saturating increment.
    c = bodies.frames_below_threshold[b] + wp.int32(1)
    if c > sleeping_frames_required:
        c = sleeping_frames_required
    bodies.frames_below_threshold[b] = c
    if c >= sleeping_frames_required:
        bodies.island_root[b] = island_root_per_compact[island]


# ---------------------------------------------------------------------------
# Wake paths: self-wake (in-step) and external input (pre-collide).
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _phoenx_self_wake_fanin_kernel(
    bodies: BodyContainer,
    body_aabb_diagonal: wp.array[wp.float32],
    sleeping_velocity_threshold: wp.float32,
    step_dt: wp.float32,
    wake_flag: wp.array[wp.int32],
):
    """Per sleeping body: if its predicted score exceeds the threshold,
    raise the wake flag for its ``island_root``. Catches host-side
    velocity (``state.body_qd``) or force injection that bypasses
    :meth:`PhoenXWorld.wake_on_external_input`. Same score formula as
    :func:`_phoenx_island_max_velocity_kernel`.
    """
    b = wp.tid()
    if bodies.motion_type[b] != MOTION_DYNAMIC:
        return
    root = bodies.island_root[b]
    if root < wp.int32(0):
        return
    inv_mass = bodies.inverse_mass[b]
    if inv_mass == 0.0:
        return
    force = bodies.force[b]
    torque = bodies.torque[b]
    v_pred = bodies.velocity[b] + force * (inv_mass * step_dt)
    w_pred = bodies.angular_velocity[b] + (mat33_from_sym6(bodies.inverse_inertia_world[b]) * torque) * step_dt
    score = wp.length(v_pred) + 0.5 * body_aabb_diagonal[b] * wp.length(w_pred)
    if wp.length_sq(force) > wp.float32(0.0) or wp.length_sq(torque) > wp.float32(0.0):
        score = wp.max(score, sleeping_velocity_threshold)
    if score >= sleeping_velocity_threshold:
        wp.atomic_max(wake_flag, root, wp.int32(1))


@wp.kernel(enable_backward=False)
def _phoenx_apply_wake_flag_kernel(
    bodies: BodyContainer,
    wake_flag: wp.array[wp.int32],
):
    """Clear ``island_root`` for every body whose root is flagged. Atomic
    across the whole root group: one body triggering wake takes the
    whole sleeping island awake in one frame."""
    b = wp.tid()
    root = bodies.island_root[b]
    if root < wp.int32(0):
        return
    if wake_flag[root] != wp.int32(0):
        bodies.island_root[b] = wp.int32(-1)


@wp.kernel(enable_backward=False)
def _phoenx_island_fanin_external_input_kernel(
    bodies: BodyContainer,
    island_has_external: wp.array[wp.int32],
):
    """Pre-collide variant of ``_phoenx_self_wake_fanin_kernel``: raises
    a per-root flag for any sleeping body carrying a user-applied force
    or torque, so :meth:`PhoenXWorld.wake_on_external_input` can wake
    the island before the broad-phase filter drops supporting contact
    pairs. Gravity is *not* counted (it lives on a separate per-world
    array and never touches ``bodies.force``)."""
    b = wp.tid()
    if bodies.motion_type[b] != MOTION_DYNAMIC:
        return
    root = bodies.island_root[b]
    if root < wp.int32(0):
        return
    if wp.length_sq(bodies.force[b]) > wp.float32(0.0) or wp.length_sq(bodies.torque[b]) > wp.float32(0.0):
        wp.atomic_max(island_has_external, root, wp.int32(1))


@wp.kernel(enable_backward=False)
def _phoenx_apply_island_wake_kernel(
    bodies: BodyContainer,
    island_has_external: wp.array[wp.int32],
):
    """Pre-collide variant of ``_phoenx_apply_wake_flag_kernel``."""
    b = wp.tid()
    root = bodies.island_root[b]
    if root < wp.int32(0):
        return
    if island_has_external[root] != wp.int32(0):
        bodies.island_root[b] = wp.int32(-1)


# ---------------------------------------------------------------------------
# Element collapse for the partitioner / colorer.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _phoenx_collapse_sleeping_elements_kernel(
    bodies: BodyContainer,
    num_active_constraints: wp.array[wp.int32],
    num_bodies: wp.int32,
    elements: wp.array[ElementInteractionData],
):
    """Rewrite every per-element body slot whose body is sleeping to -1
    so the partitioner adjacency count drops the constraint from every
    colour bucket. Particles and already-``-1`` slots pass through."""
    tid = wp.tid()
    if tid >= num_active_constraints[0]:
        return
    el = elements[tid]
    for j in range(MAX_BODIES):
        b = el.bodies[j]
        if b < 0:
            break
        if b < num_bodies and bodies.island_root[b] >= wp.int32(0):
            el.bodies[j] = wp.int32(-1)
    elements[tid] = el
