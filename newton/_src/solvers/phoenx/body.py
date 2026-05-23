"""Rigid-body SoA storage for :class:`PhoenXWorld`.

One ``wp.array`` per field, indexed by body id. All fields are float32.
"""

import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_NONE,
    ACCESS_MODE_STATIC,
    ACCESS_MODE_VELOCITY_LEVEL,
    synchronize_pose_velocity,
)

__all__ = [
    "MOTION_DYNAMIC",
    "MOTION_KINEMATIC",
    "MOTION_STATIC",
    "BodyContainer",
    "body_container_zeros",
    "body_set_access_mode",
]


# Motion types (mirrors Jitter2 ``MotionType``).
MOTION_STATIC = wp.constant(0)
MOTION_KINEMATIC = wp.constant(1)
MOTION_DYNAMIC = wp.constant(2)


@wp.struct
class BodyContainer:
    """SoA storage for a batch of rigid bodies. All arrays length ``num_bodies``."""

    position: wp.array[wp.vec3f]
    velocity: wp.array[wp.vec3f]
    angular_velocity: wp.array[wp.vec3f]

    orientation: wp.array[wp.quatf]

    #: Local-frame offset from body-origin (Newton contact-anchor frame) to COM.
    #: :attr:`position` tracks the COM; contact anchors must subtract ``body_com``
    #: before being used as lever arms about the COM.
    body_com: wp.array[wp.vec3f]

    inverse_inertia_world: wp.array[wp.mat33f]
    inverse_inertia: wp.array[wp.mat33f]

    inverse_mass: wp.array[wp.float32]

    force: wp.array[wp.vec3f]
    torque: wp.array[wp.vec3f]

    linear_damping: wp.array[wp.float32]
    angular_damping: wp.array[wp.float32]

    affected_by_gravity: wp.array[wp.int32]
    motion_type: wp.array[wp.int32]
    world_id: wp.array[wp.int32]

    # Kinematic-body pose scripting: prev = lerp/slerp origin, target = endpoint.
    # If kinematic_target_valid == 0 the prepare kernel synthesises target from
    # constant-velocity fallthrough.
    position_prev: wp.array[wp.vec3f]
    orientation_prev: wp.array[wp.quatf]
    kinematic_target_pos: wp.array[wp.vec3f]
    kinematic_target_orient: wp.array[wp.quatf]
    kinematic_target_valid: wp.array[wp.int32]

    #: Pose snapshot at substep entry. Read by
    #: :mod:`newton._src.solvers.phoenx.access_mode` synchronize helpers
    #: as the finite-diff anchor when a constraint flips this body's
    #: :attr:`access_mode` between velocity- and position-level. Written
    #: once per substep by ``_phoenx_apply_forces_and_gravity_kernel``;
    #: not read in the rigid PGS hot path.
    position_prev_substep: wp.array[wp.vec3f]
    orientation_prev_substep: wp.array[wp.quatf]
    #: Per-body access-mode tag (see
    #: :mod:`newton._src.solvers.phoenx.access_mode`). Set at substep
    #: entry to ``VELOCITY_LEVEL`` for dynamic bodies and ``STATIC`` for
    #: anchored / kinematic / pinned bodies. Constraint kernels that
    #: want to read or write at a specific level call
    #: ``body_set_access_mode`` to flip lazily.
    access_mode: wp.array[wp.int32]

    #: Single-element scene-wide flag. ``1`` if the scene contains any
    #: constraint type that can write ``ACCESS_MODE_POSITION_LEVEL``
    #: (cloth triangle, cloth bending, soft-tet XPBD) -- ``0`` otherwise.
    #: Read as the first line of :func:`body_set_access_mode` /
    #: :func:`particle_set_access_mode`: a warp-uniform broadcast load
    #: that short-circuits the whole flip in rigid-only scenes without
    #: touching the scattered :attr:`access_mode` array.
    has_position_level_writers: wp.array[wp.int32]

    #: Per-body sleep label. ``-1`` = awake; any non-negative value is
    #: the *root body id* (the lowest body id in the island this body
    #: was in at the moment it fell asleep). Acts as both the sleeping
    #: flag and the persistent island identifier consumed by
    #: :meth:`PhoenXWorld.wake_on_external_input` -- a force on any
    #: sleeping body lifts every body sharing the same ``island_root``.
    #: Only written when :attr:`PhoenXWorld.sleeping_velocity_threshold`
    #: > 0; stays at ``-1`` (awake) when sleeping is disabled. Sleeping
    #: bodies skip gravity / force application, are collapsed to -1 in
    #: the partitioner's element view (so their constraints drop out
    #: of the coloring), and are excluded from the per-step union-find
    #: island build (so the live ``set_nr`` covers awake bodies only).
    island_root: wp.array[wp.int32]

    #: Per-body counter: number of consecutive frames the body's island
    #: has had its max-velocity score below
    #: :attr:`PhoenXWorld.sleeping_velocity_threshold`. Reset to 0 the
    #: moment any body in the island exceeds the threshold (the
    #: island-level max governs every body in the island). Saturates
    #: at :attr:`PhoenXWorld.sleeping_frames_required` so it never
    #: overflows. :attr:`island_root` is stamped with the island's
    #: lowest body id once this counter reaches the required frame
    #: count; with ``sleeping_frames_required == 0`` the body sleeps
    #: on the first below-threshold frame.
    frames_below_threshold: wp.array[wp.int32]


def body_container_zeros(num_bodies: int, device: wp.DeviceLike = None) -> BodyContainer:
    """Allocate a zero-initialized :class:`BodyContainer` on ``device``.

    Defaults: damping = 1.0 (none), affected_by_gravity = 1, motion_type = STATIC,
    world_id = 0. Quaternions are zero — callers must set identity if needed.
    """
    c = BodyContainer()
    c.position = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.velocity = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.angular_velocity = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.orientation = wp.zeros(num_bodies, dtype=wp.quatf, device=device)
    c.body_com = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.inverse_inertia_world = wp.zeros(num_bodies, dtype=wp.mat33f, device=device)
    c.inverse_inertia = wp.zeros(num_bodies, dtype=wp.mat33f, device=device)
    c.inverse_mass = wp.zeros(num_bodies, dtype=wp.float32, device=device)
    c.force = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.torque = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.linear_damping = wp.full(num_bodies, value=1.0, dtype=wp.float32, device=device)
    c.angular_damping = wp.full(num_bodies, value=1.0, dtype=wp.float32, device=device)
    c.affected_by_gravity = wp.full(num_bodies, value=1, dtype=wp.int32, device=device)
    c.motion_type = wp.full(num_bodies, value=int(MOTION_STATIC), dtype=wp.int32, device=device)
    c.world_id = wp.zeros(num_bodies, dtype=wp.int32, device=device)
    c.position_prev = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.orientation_prev = wp.zeros(num_bodies, dtype=wp.quatf, device=device)
    c.kinematic_target_pos = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.kinematic_target_orient = wp.zeros(num_bodies, dtype=wp.quatf, device=device)
    c.kinematic_target_valid = wp.zeros(num_bodies, dtype=wp.int32, device=device)
    c.position_prev_substep = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.orientation_prev_substep = wp.zeros(num_bodies, dtype=wp.quatf, device=device)
    c.access_mode = wp.full(num_bodies, value=int(ACCESS_MODE_VELOCITY_LEVEL), dtype=wp.int32, device=device)
    c.has_position_level_writers = wp.zeros(1, dtype=wp.int32, device=device)
    c.island_root = wp.full(num_bodies, value=-1, dtype=wp.int32, device=device)
    c.frames_below_threshold = wp.zeros(num_bodies, dtype=wp.int32, device=device)
    return c


@wp.func
def body_set_access_mode(
    bodies: BodyContainer,
    b: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Lazy SoA wrapper around :func:`synchronize_pose_velocity`.

    First check is the warp-uniform scene-wide gate
    :attr:`BodyContainer.has_position_level_writers` -- when ``0``
    (no cloth / cloth-bend / soft-tet in the scene) the entire flip
    is provably a no-op, and the broadcast load avoids touching the
    scattered :attr:`access_mode` array. Second check is the per-body
    same-mode fast path.

    Mirrors Jitter2 ``TinyRigidState.SynchronizeVelAndPosStateUpdates``
    (``MassSplitting/TinyRigidState.cs:62-65``).
    """
    if bodies.has_position_level_writers[0] == 0:
        return
    current = bodies.access_mode[b]
    # Early-out paths -- match the no-op short-circuits in
    # :func:`synchronize_pose_velocity`. Skipping the function call
    # avoids reading the six pose / velocity arrays on a no-op flip,
    # which is the common case for rigid-only scenes (every body
    # stays in VELOCITY_LEVEL after the substep-entry kernel sets it).
    if current == new_access_mode:
        return
    if current == ACCESS_MODE_STATIC:
        return
    if current == ACCESS_MODE_NONE:
        bodies.access_mode[b] = new_access_mode
        return
    p_new, q_new, v_new, w_new, mode_new = synchronize_pose_velocity(
        bodies.position[b],
        bodies.orientation[b],
        bodies.velocity[b],
        bodies.angular_velocity[b],
        bodies.position_prev_substep[b],
        bodies.orientation_prev_substep[b],
        current,
        new_access_mode,
        inv_dt,
    )
    bodies.position[b] = p_new
    bodies.orientation[b] = q_new
    bodies.velocity[b] = v_new
    bodies.angular_velocity[b] = w_new
    bodies.access_mode[b] = mode_new
