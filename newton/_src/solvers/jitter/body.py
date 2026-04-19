"""Rigid-body data for the Jitter port.

Mirrors the relevant subset of ``Jitter2.Dynamics.RigidBodyData`` (see
``C:/git3/jitterphysics2/src/Jitter2/Dynamics/RigidBody.cs``). Only the
fields that the constraint solver actually touches are included today;
add more (forces, deltas, friction, ...) as additional Jitter modules are
ported.

In Jitter, ``JHandle<RigidBodyData>`` indirects into a ``PartitionedBuffer``
(an array-of-structs). In this port we replace the handle with a plain
``int`` body id and store the bodies in a :class:`BodyContainer`, which is
**struct-of-arrays** (one ``wp.array`` per field). On the GPU this gives:

* coalesced loads/stores when many threads in a warp touch the same
  field (each field array is contiguous, so 32 lanes' worth of, say,
  ``position`` fall into one or two 128-byte transactions);
* cheap partial reads when a kernel only needs a subset of fields (e.g.
  an integration kernel that only touches position/velocity does not
  pull inertia tensors through the cache);
* trivial extension when new per-body quantities are added (just append a
  field to ``BodyContainer`` -- no padding/alignment changes for the rest).

The trade-off is that gathering a *full* :class:`RigidBodyData` value
(which the constraint kernels do per thread for body1/body2) costs one
load per field. Constraints touch ~6 fields each, scattered by the graph
colorer, so neither layout coalesces perfectly; SoA still wins because of
the partial-access benefit and because each field load is a contiguous
16-byte (vec3/quat) or 36-byte (mat33) read instead of a strided ~96-byte
AoS struct load.

Use :func:`body_container_get` / :func:`body_container_set` inside
``@wp.func`` / ``@wp.kernel`` code to pack/unpack a single body, and
:func:`body_container_zeros` from the host to allocate a container.

All physical quantities are stored in single precision (``float32``),
matching the rest of Newton's solver stack.
"""

import warp as wp

__all__ = [
    "BodyContainer",
    "RigidBodyData",
    "body_container_get",
    "body_container_set",
    "body_container_zeros",
]


@wp.struct
class RigidBodyData:
    """Per-body state required by the constraint kernels.

    This is the *value type* used inside ``@wp.func`` / ``@wp.kernel``
    code: kernels gather a full :class:`RigidBodyData` from a
    :class:`BodyContainer`, work with it locally, then scatter it back.

    Field names use Newton's snake_case convention; the mapping to the
    Jitter ``RigidBodyData`` field names is one-for-one:

    * ``position``               <-> ``Position``
    * ``velocity``               <-> ``Velocity``
    * ``angular_velocity``       <-> ``AngularVelocity``
    * ``orientation``            <-> ``Orientation``
    * ``inverse_inertia_world``  <-> ``InverseInertiaWorld``
    * ``inverse_mass``           <-> ``InverseMass``
    """

    position: wp.vec3f
    velocity: wp.vec3f
    angular_velocity: wp.vec3f

    orientation: wp.quatf

    inverse_inertia_world: wp.mat33f

    inverse_mass: wp.float32


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

    inverse_inertia_world: wp.array[wp.mat33f]

    inverse_mass: wp.array[wp.float32]


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
    b.inverse_inertia_world = c.inverse_inertia_world[i]
    b.inverse_mass = c.inverse_mass[i]
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
    c.inverse_inertia_world[i] = b.inverse_inertia_world
    c.inverse_mass[i] = b.inverse_mass


def body_container_zeros(num_bodies: int, device: wp.DeviceLike = None) -> BodyContainer:
    """Allocate a zero-initialized :class:`BodyContainer` for ``num_bodies``
    bodies on ``device``.

    Quaternions are zero-initialized too (i.e. ``(0, 0, 0, 0)``); callers
    that need identity rotations should overwrite the ``orientation``
    array, e.g. ``c.orientation.fill_(wp.quatf(0.0, 0.0, 0.0, 1.0))``.
    Same goes for inertia tensors / inverse masses, which the user is
    responsible for filling in before any solver step.
    """
    c = BodyContainer()
    c.position = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.velocity = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.angular_velocity = wp.zeros(num_bodies, dtype=wp.vec3f, device=device)
    c.orientation = wp.zeros(num_bodies, dtype=wp.quatf, device=device)
    c.inverse_inertia_world = wp.zeros(num_bodies, dtype=wp.mat33f, device=device)
    c.inverse_mass = wp.zeros(num_bodies, dtype=wp.float32, device=device)
    return c
