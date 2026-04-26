"""Rigid-body SoA storage for :class:`PhoenXWorld`.

Struct-of-arrays: one ``wp.array`` per field, indexed by body id. SoA
gives coalesced per-field loads (32 lanes -> 1-2 128-byte transactions)
and cheap partial reads when a kernel only touches a subset of fields.
Kernels access fields directly (``bodies.position[i]`` etc.) -- there is
no per-body gather/scatter helper because the solver-side kernels
typically read only a handful of fields and the compiler already keeps
them in registers across the inner constraint loop.
:func:`body_container_zeros` on the host allocates the SoA.
All quantities are single precision (``float32``).
"""

import warp as wp

__all__ = [
    "MOTION_DYNAMIC",
    "MOTION_KINEMATIC",
    "MOTION_STATIC",
    "BodyContainer",
    "body_container_zeros",
]


# Motion types (mirrors Jitter2 ``MotionType``). Stored as int32 in the
# container -- Warp structs can't hold Python enums, and we want this in
# the constraint-solver hot path.
MOTION_STATIC = wp.constant(0)
MOTION_KINEMATIC = wp.constant(1)
MOTION_DYNAMIC = wp.constant(2)


@wp.struct
class BodyContainer:
    """Struct-of-arrays storage for a batch of rigid bodies.

    Each field is a 1-D ``wp.array`` of length ``num_bodies`` (must
    match across fields). Constructed on the host with
    :func:`body_container_zeros`; kernels read/write the fields
    directly, e.g. ``bodies.position[i]``.
    """

    position: wp.array[wp.vec3f]
    velocity: wp.array[wp.vec3f]
    angular_velocity: wp.array[wp.vec3f]

    orientation: wp.array[wp.quatf]

    #: Per-body local-frame offset from the body-origin frame (the frame
    #: Newton's narrow phase expresses contact anchors in) to the body's
    #: centre of mass. :attr:`position` tracks the *COM* in world space,
    #: so contact anchors must subtract ``body_com`` before being used as
    #: lever arms about the COM. Zero for shapes whose mesh origin
    #: coincides with the COM (boxes, spheres); non-zero for asymmetric
    #: meshes (bunny, nut, etc.).
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

    # --- Kinematic-body pose scripting ---------------------------------
    #
    # :data:`MOTION_KINEMATIC` bodies take a per-frame scripted pose
    # (via :meth:`PhoenXWorld.set_kinematic_pose` or ``state.body_q``).
    # The solver infers linear + angular velocity from the pose delta
    # and lerp/slerp-interpolates across substeps so contacts see
    # smooth motion. With no scripted target this step
    # (``kinematic_target_valid == 0``), the prepare kernel synthesises
    # ``pos_prev + velocity * dt`` as the target for constant-velocity
    # fallthrough.

    #: End-of-previous-step pose (lerp/slerp origin). Written once at
    #: :meth:`PhoenXWorld.step` entry by the kinematic prepare kernel.
    position_prev: wp.array[wp.vec3f]
    orientation_prev: wp.array[wp.quatf]

    #: End-of-current-step pose (lerp/slerp endpoint). Written by the
    #: user via :meth:`PhoenXWorld.set_kinematic_pose` or by the
    #: Newton adapter; the solver never overwrites it after the
    #: prepare kernel resolves it.
    kinematic_target_pos: wp.array[wp.vec3f]
    kinematic_target_orient: wp.array[wp.quatf]

    #: 1 = ``kinematic_target_*`` holds a user-supplied pose to use
    #: this step; 0 = no explicit target, so the prepare kernel
    #: synthesises one from the constant-velocity path. Reset to 0
    #: by prepare so the user must re-assert the target each step
    #: (either explicitly or by re-importing Newton state).
    kinematic_target_valid: wp.array[wp.int32]


def body_container_zeros(num_bodies: int, device: wp.DeviceLike = None) -> BodyContainer:
    """Allocate a zero-initialized :class:`BodyContainer` for ``num_bodies``
    bodies on ``device``.

    Defaults match Jitter2's per-body defaults so callers only have to
    fill in what's actually non-default:

    * Quaternions zero-initialised -- callers that want identity
      rotations should overwrite ``orientation`` (e.g. with
      ``wp.quatf(0, 0, 0, 1)``).
    * ``linear_damping`` / ``angular_damping`` = 1.0 (no damping).
    * ``affected_by_gravity`` = 1.
    * ``motion_type`` = ``MOTION_STATIC`` -- callers must set dynamic
      bodies explicitly. (We default to static so a forgotten
      initialisation produces obviously-frozen objects rather than
      silently-broken dynamics with zero inertia.)
    """
    c = BodyContainer()
    c.position = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.velocity = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.angular_velocity = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.orientation = wp.zeros(num_bodies, dtype=wp.quatf, device=device)
    # body_com defaults to zero -- safe for symmetric primitives (boxes,
    # spheres). Non-zero meshes (bunny, nut, etc.) must overwrite this
    # from ``model.body_com`` via the solver's init / sync kernels so
    # the contact lever arm maths in :mod:`constraint_contact` uses the
    # correct origin-to-COM offset.
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
    # world_id defaults to 0 -- single-world scenes leave this
    # untouched and the multi-world dispatcher collapses to the same
    # single-block behaviour as before the refactor.
    c.world_id = wp.zeros(num_bodies, dtype=wp.int32, device=device)
    # Kinematic-body pose-scripting scratch. ``position_prev`` /
    # ``orientation_prev`` are initialised to zero; the host-side
    # packer (:func:`WorldBuilder._build_body_container` or the Newton
    # adapter) must overwrite them with the body's initial pose so
    # the first :meth:`PhoenXWorld.step` sees ``prev == target`` for
    # kinematic bodies that the user hasn't touched (velocity = 0).
    c.position_prev = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.orientation_prev = wp.zeros(num_bodies, dtype=wp.quatf, device=device)
    c.kinematic_target_pos = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.kinematic_target_orient = wp.zeros(num_bodies, dtype=wp.quatf, device=device)
    c.kinematic_target_valid = wp.zeros(num_bodies, dtype=wp.int32, device=device)
    return c
