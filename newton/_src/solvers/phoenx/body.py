"""Rigid-body SoA storage for :class:`PhoenXWorld`.

One ``wp.array`` per field, indexed by body id. All fields are float32.
"""

import warp as wp

__all__ = [
    "MOTION_DYNAMIC",
    "MOTION_KINEMATIC",
    "MOTION_STATIC",
    "BodyContainer",
    "body_container_zeros",
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
    return c
