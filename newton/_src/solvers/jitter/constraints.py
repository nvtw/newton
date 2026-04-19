"""Warp port of Jitter2's BallSocket constraint.

Direct translation of ``Jitter2.Dynamics.Constraints.BallSocket``
(``C:/git3/jitterphysics2/src/Jitter2/Dynamics/Constraints/BallSocket.cs``).

The C# code uses ``JHandle<RigidBodyData>`` to indirect into a partitioned
buffer; here a body handle is just an ``int32`` index into a
:class:`BodyContainer` (struct-of-arrays storage, see
:mod:`newton._src.solvers.jitter.body`). Constraint kernels and ``@wp.func``
helpers therefore take that container as an additional parameter and
read/write the relevant entries via :func:`body_container_get` /
:func:`body_container_set`.

Concurrency: the constraint kernels assume the launcher has partitioned
the constraint set so that no two constraints in the same launch touch
the same body (this is what
:class:`newton._src.solvers.jitter.graph_coloring_incremental.IncrementalContactPartitioner`
produces). Within a partition each thread can therefore do a plain
read-modify-write of body ``body1`` / ``body2`` without atomics.

Mapping summary:

* ``JVector``                            -> ``wp.vec3f``
* ``JQuaternion``                        -> ``wp.quatf``
* ``JMatrix``                            -> ``wp.mat33f``
* ``JVector.Transform(v, M)``            -> ``M @ v`` (matrix-vector)
* ``JVector.Transform(v, q)``            -> ``wp.quat_rotate(q, v)``
* ``JVector.ConjugatedTransform(v, q)``  -> ``wp.quat_rotate_inv(q, v)``
* ``JMatrix.CreateCrossProduct(r)``      -> ``wp.skew(r)``
* ``JMatrix.MultiplyTransposed(A, B)``   -> ``A @ wp.transpose(B)``
* ``JMatrix.Identity``                   -> ``wp.identity(3, dtype=...)``
* ``JMatrix.Inverse(M, out M)``          -> ``wp.inverse(M)``
"""

import warp as wp

from newton._src.solvers.jitter.body import (
    BodyContainer,
    RigidBodyData,
)

__all__ = [
    "DEFAULT_LINEAR_BIAS",
    "DEFAULT_LINEAR_SOFTNESS",
    "BallSocketData",
    "ball_socket_initialize",
    "ball_socket_iterate_kernel",
    "ball_socket_prepare_for_iteration_kernel",
]


# ---------------------------------------------------------------------------
# Constants (mirrors Jitter2.Dynamics.Constraints.Constraint)
# ---------------------------------------------------------------------------

DEFAULT_LINEAR_SOFTNESS = wp.constant(wp.float32(0.00001))
DEFAULT_LINEAR_BIAS = wp.constant(wp.float32(0.2))


# ---------------------------------------------------------------------------
# Constraint state
# ---------------------------------------------------------------------------


@wp.struct
class BallSocketData:
    """Per-constraint state for a ball-and-socket joint.

    Mirrors ``BallSocket.BallSocketData`` field-for-field. The C# struct
    embeds ``JHandle<RigidBodyData>`` for each body; we replace that with
    a plain integer index into the solver's body array.

    Field order matches the C# struct so future debug dumps line up; the
    layout itself is irrelevant to Warp.
    """

    # Internal/dispatch fields from the C# layout. Kept so we can debug
    # against Jitter byte-for-byte if needed; not consumed by the solver.
    # TODO(jitter-port): drop ``_internal`` / ``dispatch_id`` /
    # ``constraint_id`` once the Warp Jitter solver is wired up and we
    # confirm none of them are needed (Jitter uses them for the partitioned
    # buffer indirection and the function-pointer dispatch table, neither of
    # which we replicate here).
    _internal: wp.int32
    dispatch_id: wp.uint32
    constraint_id: wp.uint64

    # Body indices (replaces JHandle<RigidBodyData>).
    body1: wp.int32
    body2: wp.int32

    # Anchor points expressed in each body's local frame.
    local_anchor1: wp.vec3f
    local_anchor2: wp.vec3f

    # Scratch / cached values populated each substep.
    u: wp.vec3f
    r1: wp.vec3f
    r2: wp.vec3f

    bias_factor: wp.float32
    softness: wp.float32

    effective_mass: wp.mat33f
    accumulated_impulse: wp.vec3f

    # NB: Jitter calls the cached vec3 RHS field ``Bias``; we use the
    # explicit ``bias`` name here. The scalar ``bias_factor`` above is
    # what Jitter exposes via the ``Bias`` *property* (which returns the
    # scalar BiasFactor field).
    bias: wp.vec3f


# ---------------------------------------------------------------------------
# Initialization (host-side helper, mirrors BallSocket.Initialize)
# ---------------------------------------------------------------------------


def ball_socket_initialize(
    data: BallSocketData,
    body1: RigidBodyData,
    body2: RigidBodyData,
    anchor: wp.vec3f,
) -> BallSocketData:
    """Compute local anchor points from a shared world-space anchor.

    Direct port of ``BallSocket.Initialize`` (BallSocket.cs:65-80). Runs
    on the host once at constraint creation time (the user picked the
    "host helper" option); for batched setup, write the same arithmetic
    in a small ``@wp.kernel``.

    Args:
        data: The constraint to initialize. Must already have ``body1``
            and ``body2`` set.
        body1: A copy of the first body's state at construction time.
        body2: A copy of the second body's state at construction time.
        anchor: The shared anchor point in world space [m].

    Returns:
        A new :class:`BallSocketData` with the anchor / softness / bias
        fields populated. The other fields are unchanged.
    """
    # JVector.Subtract(anchor, body1.Position, out la1)
    la1 = anchor - body1.position
    la2 = anchor - body2.position

    # JVector.ConjugatedTransform(la, body.Orientation, out la)  -- inverse rotation.
    la1 = wp.quat_rotate_inv(body1.orientation, la1)
    la2 = wp.quat_rotate_inv(body2.orientation, la2)

    data.local_anchor1 = la1
    data.local_anchor2 = la2
    data.bias_factor = float(DEFAULT_LINEAR_BIAS)
    data.softness = float(DEFAULT_LINEAR_SOFTNESS)
    return data


# ---------------------------------------------------------------------------
# Per-iteration math (wp.func helpers + dispatch kernels)
# ---------------------------------------------------------------------------


@wp.func
def _ball_socket_prepare_for_iteration(
    data: BallSocketData,
    bodies: BodyContainer,
    idt: wp.float32,
) -> BallSocketData:
    """Direct port of ``PrepareForIterationBallSocket`` (BallSocket.cs:130).

    Reads ``data.body1`` / ``data.body2`` to look up the two bodies in the
    solver's :class:`BodyContainer`, recomputes the cached lever arms /
    effective mass / bias, and applies warm-start impulses to both bodies.
    Only the fields actually consumed are loaded from ``bodies`` and only
    ``velocity`` / ``angular_velocity`` are written back.
    """
    b1 = data.body1
    b2 = data.body2

    # Load only what we need: orientation + position for the lever arms,
    # inverse mass + inverse world-inertia for the effective mass and warm
    # start, velocities for the warm-start update.
    orientation1 = bodies.orientation[b1]
    orientation2 = bodies.orientation[b2]
    position1 = bodies.position[b1]
    position2 = bodies.position[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    # JVector.Transform(LocalAnchor, body.Orientation, out R)
    data.r1 = wp.quat_rotate(orientation1, data.local_anchor1)
    data.r2 = wp.quat_rotate(orientation2, data.local_anchor2)

    # World-space anchor positions on each body.
    p1 = position1 + data.r1
    p2 = position2 + data.r2

    # cr_i = [r_i]_x   (skew-symmetric cross-product matrix)
    cr1 = wp.skew(data.r1)
    cr2 = wp.skew(data.r2)

    # EffectiveMass = m1^-1 * I + cr1 * (InvI1 * cr1^T)
    #               + m2^-1 * I + cr2 * (InvI2 * cr2^T)
    # JMatrix.MultiplyTransposed(A, B) == A * B^T in Jitter, so we mirror
    # that explicitly with wp.transpose to keep the translation 1:1.
    eye3 = wp.identity(3, dtype=wp.float32)
    eff = inv_mass1 * eye3
    eff = eff + cr1 @ (inv_inertia1 @ wp.transpose(cr1))
    eff = eff + inv_mass2 * eye3
    eff = eff + cr2 @ (inv_inertia2 @ wp.transpose(cr2))

    # Softness contribution: add softness * idt to the diagonal.
    softness = data.softness * idt
    eff[0, 0] = eff[0, 0] + softness
    eff[1, 1] = eff[1, 1] + softness
    eff[2, 2] = eff[2, 2] + softness

    data.effective_mass = wp.inverse(eff)

    # Positional bias drives p1 and p2 back together over the next step.
    data.bias = (p2 - p1) * data.bias_factor * idt

    # Warm start: re-apply the previous solve's accumulated impulse.
    acc = data.accumulated_impulse

    velocity1 = bodies.velocity[b1] - inv_mass1 * acc
    # AngularVelocity -= InvI * (cr * acc)
    angular_velocity1 = bodies.angular_velocity[b1] - inv_inertia1 @ (cr1 @ acc)

    velocity2 = bodies.velocity[b2] + inv_mass2 * acc
    angular_velocity2 = bodies.angular_velocity[b2] + inv_inertia2 @ (cr2 @ acc)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2

    return data


@wp.func
def _ball_socket_iterate(
    data: BallSocketData,
    bodies: BodyContainer,
    idt: wp.float32,
) -> BallSocketData:
    """Direct port of ``IterateBallSocket`` (BallSocket.cs:199).

    One PGS-style iteration: build the constraint Jacobian-times-velocity,
    project to find the corrective impulse ``lambda``, accumulate it, and
    apply the matching velocity / angular-velocity deltas to the two bodies.
    Only velocity / angular-velocity / inverse mass / inverse world-inertia
    are touched -- positions and orientations are not needed here.
    """
    b1 = data.body1
    b2 = data.body2

    velocity1 = bodies.velocity[b1]
    velocity2 = bodies.velocity[b2]
    angular_velocity1 = bodies.angular_velocity[b1]
    angular_velocity2 = bodies.angular_velocity[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    cr1 = wp.skew(data.r1)
    cr2 = wp.skew(data.r2)

    # softnessVector = AccumulatedImpulse * Softness * idt
    softness_vector = data.accumulated_impulse * data.softness * idt

    # jv = -v1 + cr1 * w1 + v2 - cr2 * w2
    jv = -velocity1 + cr1 @ angular_velocity1 + velocity2 - cr2 @ angular_velocity2

    # lambda = -EffectiveMass * (jv + Bias + softnessVector)
    lam = -(data.effective_mass @ (jv + data.bias + softness_vector))

    data.accumulated_impulse = data.accumulated_impulse + lam

    bodies.velocity[b1] = velocity1 - inv_mass1 * lam
    bodies.angular_velocity[b1] = angular_velocity1 - inv_inertia1 @ (cr1 @ lam)

    bodies.velocity[b2] = velocity2 + inv_mass2 * lam
    bodies.angular_velocity[b2] = angular_velocity2 + inv_inertia2 @ (cr2 @ lam)

    return data


@wp.kernel
def ball_socket_prepare_for_iteration_kernel(
    constraints: wp.array[BallSocketData],
    bodies: BodyContainer,
    idt: wp.float32,
    partition_element_ids: wp.array[int],
    partition_count: wp.array[int],
):
    """Indirect launch over a single graph-coloring partition.

    The launch is sized by the constraint *capacity*; each thread reads
    ``partition_element_ids[tid]`` to find the actual constraint id it
    should process. Threads with ``tid >= partition_count[0]`` early-out.

    The partitioner guarantees that no two constraints in the same
    partition share a body, so the per-thread read-modify-write of
    ``bodies`` is race-free.
    """
    tid = wp.tid()
    if tid >= partition_count[0]:
        return
    cid = partition_element_ids[tid]
    data = constraints[cid]
    data = _ball_socket_prepare_for_iteration(data, bodies, idt)
    constraints[cid] = data


@wp.kernel
def ball_socket_iterate_kernel(
    constraints: wp.array[BallSocketData],
    bodies: BodyContainer,
    idt: wp.float32,
    partition_element_ids: wp.array[int],
    partition_count: wp.array[int],
):
    """Indirect launch over a single graph-coloring partition. See
    :func:`ball_socket_prepare_for_iteration_kernel` for the contract."""
    tid = wp.tid()
    if tid >= partition_count[0]:
        return
    cid = partition_element_ids[tid]
    data = constraints[cid]
    data = _ball_socket_iterate(data, bodies, idt)
    constraints[cid] = data
