"""Rigid-body SoA storage for :class:`PhoenXWorld`.

Struct-of-arrays: one ``wp.array`` per field, indexed by body id. SoA
gives coalesced per-field loads (32 lanes -> 1-2 128-byte transactions)
and cheap partial reads when a kernel only touches a subset of fields.
Use :func:`body_container_get` / :func:`body_container_set` inside
``@wp.func``/``@wp.kernel`` to pack/unpack a single body;
:func:`body_container_zeros` on the host to allocate.
All quantities are single precision (``float32``).
"""

import warp as wp

__all__ = [
    "MOTION_DYNAMIC",
    "MOTION_KINEMATIC",
    "MOTION_STATIC",
    "BodyContainer",
    "RigidBodyData",
    "body_container_get",
    "body_container_set",
    "body_container_zeros",
]


# Motion types (mirrors Jitter2 ``MotionType``). Stored as int32 in the
# container -- Warp structs can't hold Python enums, and we want this in
# the constraint-solver hot path.
MOTION_STATIC = wp.constant(0)
MOTION_KINEMATIC = wp.constant(1)
MOTION_DYNAMIC = wp.constant(2)


@wp.struct
class RigidBodyData:
    """Per-body state required by the constraint kernels.

    Mirrors the relevant subset of Jitter2's ``RigidBodyData``. Naming is
    one-for-one in snake_case:

    Velocity-level state (read & written every constraint iteration):
        * ``position``               <-> ``Position``
        * ``velocity``               <-> ``Velocity``
        * ``angular_velocity``       <-> ``AngularVelocity``
        * ``orientation``            <-> ``Orientation``

    Mass / inertia (rebuilt once per step from the body-frame inertia):
        * ``inverse_inertia_world``  <-> ``InverseInertiaWorld`` (rotated)
        * ``inverse_inertia``        <-> ``inverseInertia`` (body frame, constant)
        * ``inverse_mass``           <-> ``InverseMass``

    Per-step force accumulators + cached per-substep deltas (Jitter's
    two-stage IntegrateForces split):
        * ``force``                  <-> ``Force`` (cleared in _update_bodies)
        * ``torque``                 <-> ``Torque`` (cleared in _update_bodies)
        * ``delta_velocity``         <-> ``DeltaVelocity``
        * ``delta_angular_velocity`` <-> ``DeltaAngularVelocity``

    Damping + flags:
        * ``linear_damping``         per-body multiplier applied in _update_bodies
        * ``angular_damping``        per-body multiplier applied in _update_bodies
        * ``affected_by_gravity``    0/1; gated in _update_bodies
        * ``motion_type``            see ``MOTION_*`` constants

    Multi-world:
        * ``world_id``               per-body world index (0 for
          single-world scenes). Used by :func:`_update_bodies_kernel`
          to pick the per-world gravity vector and (later) by the
          constraint dispatcher to pick the per-world CSR slice.
    """

    position: wp.vec3f
    velocity: wp.vec3f
    angular_velocity: wp.vec3f

    orientation: wp.quatf

    #: Body-local offset from the body-origin frame (the frame Newton's
    #: narrow phase expresses contact anchors in) to the body's centre
    #: of mass. :attr:`position` tracks the *COM* in world space, so
    #: when a contact anchor from the narrow phase is used as a lever
    #: arm about the COM the kernel must subtract ``body_com`` before
    #: adding it to :attr:`position`. Zero for shapes whose mesh origin
    #: coincides with the COM (boxes, spheres); non-zero for asymmetric
    #: meshes (bunny, nut, etc.).
    body_com: wp.vec3f

    inverse_inertia_world: wp.mat33f
    inverse_inertia: wp.mat33f

    inverse_mass: wp.float32

    force: wp.vec3f
    torque: wp.vec3f
    delta_velocity: wp.vec3f
    delta_angular_velocity: wp.vec3f

    linear_damping: wp.float32
    angular_damping: wp.float32

    affected_by_gravity: wp.int32
    motion_type: wp.int32
    world_id: wp.int32


@wp.struct
class BodyContainer:
    """Struct-of-arrays storage for a batch of rigid bodies.

    Each field is a 1-D ``wp.array`` of length ``num_bodies`` (must match
    across fields). Constructed on the host with
    :func:`body_container_zeros`; accessed inside kernels via
    :func:`body_container_get` and :func:`body_container_set`.

    The field set mirrors :class:`RigidBodyData` exactly so the gather /
    scatter helpers stay one-line per field.
    """

    position: wp.array[wp.vec3f]
    velocity: wp.array[wp.vec3f]
    angular_velocity: wp.array[wp.vec3f]

    orientation: wp.array[wp.quatf]

    #: Per-body local-frame offset from the body-origin frame to the
    #: COM. See :attr:`RigidBodyData.body_com` for the convention
    #: rationale.
    body_com: wp.array[wp.vec3f]

    inverse_inertia_world: wp.array[wp.mat33f]
    inverse_inertia: wp.array[wp.mat33f]

    inverse_mass: wp.array[wp.float32]

    force: wp.array[wp.vec3f]
    torque: wp.array[wp.vec3f]
    delta_velocity: wp.array[wp.vec3f]
    delta_angular_velocity: wp.array[wp.vec3f]

    linear_damping: wp.array[wp.float32]
    angular_damping: wp.array[wp.float32]

    affected_by_gravity: wp.array[wp.int32]
    motion_type: wp.array[wp.int32]
    world_id: wp.array[wp.int32]


@wp.func
def body_container_get(c: BodyContainer, i: wp.int32) -> RigidBodyData:
    """Gather the body at index ``i`` into a :class:`RigidBodyData` value.

    Use this at the top of a constraint/integration kernel to pull a
    single body into registers, then operate on the value type and write
    it back with :func:`body_container_set`.
    """
    b = RigidBodyData()
    b.position = c.position[i]
    b.velocity = c.velocity[i]
    b.angular_velocity = c.angular_velocity[i]
    b.orientation = c.orientation[i]
    b.body_com = c.body_com[i]
    b.inverse_inertia_world = c.inverse_inertia_world[i]
    b.inverse_inertia = c.inverse_inertia[i]
    b.inverse_mass = c.inverse_mass[i]
    b.force = c.force[i]
    b.torque = c.torque[i]
    b.delta_velocity = c.delta_velocity[i]
    b.delta_angular_velocity = c.delta_angular_velocity[i]
    b.linear_damping = c.linear_damping[i]
    b.angular_damping = c.angular_damping[i]
    b.affected_by_gravity = c.affected_by_gravity[i]
    b.motion_type = c.motion_type[i]
    b.world_id = c.world_id[i]
    return b


@wp.func
def body_container_set(c: BodyContainer, i: wp.int32, b: RigidBodyData):
    """Scatter a :class:`RigidBodyData` value into the container at index
    ``i``.

    The launcher must guarantee that no two threads write the same
    ``i`` concurrently (the graph-coloring partitioner provides this for
    the constraint solver); the writes are plain stores with no
    atomics."""
    c.position[i] = b.position
    c.velocity[i] = b.velocity
    c.angular_velocity[i] = b.angular_velocity
    c.orientation[i] = b.orientation
    c.body_com[i] = b.body_com
    c.inverse_inertia_world[i] = b.inverse_inertia_world
    c.inverse_inertia[i] = b.inverse_inertia
    c.inverse_mass[i] = b.inverse_mass
    c.force[i] = b.force
    c.torque[i] = b.torque
    c.delta_velocity[i] = b.delta_velocity
    c.delta_angular_velocity[i] = b.delta_angular_velocity
    c.linear_damping[i] = b.linear_damping
    c.angular_damping[i] = b.angular_damping
    c.affected_by_gravity[i] = b.affected_by_gravity
    c.motion_type[i] = b.motion_type
    c.world_id[i] = b.world_id


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
    c.delta_velocity = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.delta_angular_velocity = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.linear_damping = wp.full(num_bodies, value=1.0, dtype=wp.float32, device=device)
    c.angular_damping = wp.full(num_bodies, value=1.0, dtype=wp.float32, device=device)
    c.affected_by_gravity = wp.full(num_bodies, value=1, dtype=wp.int32, device=device)
    c.motion_type = wp.full(num_bodies, value=int(MOTION_STATIC), dtype=wp.int32, device=device)
    # world_id defaults to 0 -- single-world scenes leave this
    # untouched and the multi-world dispatcher collapses to the same
    # single-block behaviour as before the refactor.
    c.world_id = wp.zeros(num_bodies, dtype=wp.int32, device=device)
    return c
