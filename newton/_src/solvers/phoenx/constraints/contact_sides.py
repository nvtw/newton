# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-side fetch / apply helpers for the unified contact constraint.

The unified contact iterate (in
:mod:`newton._src.solvers.phoenx.constraints.constraint_contact`)
is parameterised over a (side_a_kind, side_b_kind) pair where each
kind is one of:

* :data:`~newton._src.solvers.phoenx.constraints.constraint_contact.CONTACT_KIND_RIGID`
  -- one rigid body endpoint (legacy rigid-rigid case).
* :data:`~newton._src.solvers.phoenx.constraints.constraint_contact.CONTACT_KIND_TRIANGLE`
  -- three particle endpoints + barycentric weights (cloth triangle).

The per-row PGS math (normal, two-tangent Coulomb cone, soft PD,
sticky-break, warm-start) is *side-kind-agnostic*: it reads the
contact-point world position / velocity / lever arm and the side's
contribution to ``J M^-1 J^T`` from a small :class:`SideKinematics`
struct, scatters the impulse back via a side-specific ``apply_*``
helper. The two helper families in this module are the only places
where the rigid-vs-triangle distinction shows up; everything
downstream stays uniform.

## Mass-projection rationale (triangle side)

For a contact point at barycentric coordinates ``(w_a, w_b, w_c)``
on a triangle whose three particles have inverse masses
``inv_m_{a,b,c}``, the contact-point velocity is the same weighted
combination, ``v_p = w_a v_a + w_b v_b + w_c v_c``. The diagonal
effective inverse mass for an impulse ``J`` applied at the contact
point and *scattered* via ``delta_v_i = w_i * inv_m_i * J`` is

    inv_m_eff = w_a^2 inv_m_a + w_b^2 inv_m_b + w_c^2 inv_m_c

This is the Galerkin projection of a point load onto the FEM linear
basis -- the right answer for FEM cloth, and the only choice that
keeps the iterate consistent with the cloth elasticity rows in the
same Gauss-Seidel sweep. A pinned node (``inv_m_i = 0``) drops out
correctly; a fully-pinned triangle has ``inv_m_eff = 0`` and is
treated like a static body by :func:`effective_mass_scalar`.

## Lever arm semantics

For a rigid endpoint, ``r = world_contact_point - body_world_position``;
the per-row math uses ``r`` to form ``rc = cross(r, axis)`` in the
``J M^-1 J^T`` denominator and ``cross(r, imp)`` in the angular
impulse update.

For a triangle endpoint, there is no rigid-body angular DoF: the
"angular" contribution is replaced by the cross-correlation between
nodes that the barycentric weights induce, which is *already
absorbed* into ``inv_m_eff`` via the ``sum w_i^2 inv_m_i`` formula
above. The :class:`SideKinematics` struct accordingly carries
``r = vec3(0)`` and ``inv_inertia = mat33(0)`` for triangle sides;
:func:`~newton._src.solvers.phoenx.helpers.math_helpers.effective_mass_scalar`
collapses those to give the right scalar.

## Velocity-level uniformly via the access-mode store

Both rigid and triangle sides operate at velocity-level inside the
contact iterate. Reads / writes go through the unified
:func:`~newton._src.solvers.phoenx.body_or_particle.get_velocity`
/ :func:`~newton._src.solvers.phoenx.body_or_particle.set_velocity`
accessors on :class:`BodyOrParticleStore` -- those already
encapsulate the body-vs-particle dispatch, and the per-substep
:func:`~newton._src.solvers.phoenx.particle.particle_predict_position`
/ :func:`~newton._src.solvers.phoenx.particle.particle_recover_velocity`
helpers handle the velocity ↔ position translation at the substep
entry / exit boundaries. The contact iterate doesn't need to know
that cloth particles use a position-level XPBD pass for their
elasticity rows; it just reads / writes the velocity field on each
endpoint and lets the substep boundary kernels reconcile.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.body_or_particle import (
    BodyOrParticleStore,
    get_position,
    get_velocity,
    is_particle,
    set_velocity,
)

__all__ = [
    "RigidColumnState",
    "SideKinematics",
    "SidePose",
    "apply_rigid_side",
    "apply_rigid_side_batched",
    "apply_triangle_side",
    "fetch_rigid_pose",
    "fetch_rigid_side",
    "fetch_triangle_pose",
    "fetch_triangle_side",
    "read_rigid_column_state",
    "triangle_anchor_world",
    "write_rigid_column_velocity",
]


# ---------------------------------------------------------------------------
# Per-side data carriers
# ---------------------------------------------------------------------------


@wp.struct
class SidePose:
    """Side-uniform pose data needed by :func:`_make_contact_prepare_at`.

    The prepare path needs the side's world position / orientation /
    COM offset so it can project the per-contact body-local anchor
    ``local_p`` back into world space (the lever arm ``r`` and the
    sticky-break drift derive from that projection). For triangle
    sides the "anchor world position" is the barycentric-weighted sum
    of the three particle world positions; orientation collapses to
    identity and ``body_com`` to zero.
    """

    #: World-space position of the side's representative point. For
    #: rigid sides this is the body's COM (``bodies.position[b]``);
    #: for triangle sides it's the weighted-sum world position
    #: ``w_a * x_a + w_b * x_b + w_c * x_c`` of the three triangle
    #: particles. The per-contact lever arm builds on top via
    #: ``r = world_contact_point - position`` (rigid) or ``r = 0``
    #: (triangle, see :class:`SideKinematics`).
    position: wp.vec3f
    #: World-space orientation of the side. Identity for triangle
    #: sides (no rigid DoF).
    orientation: wp.quatf
    #: COM offset in body-origin frame. Zero for triangle sides.
    body_com: wp.vec3f
    #: Diagonal inverse mass of the side at the contact point. For
    #: rigid sides this is ``bodies.inverse_mass[b]``; for triangle
    #: sides this is ``sum_i w_i^2 * inv_m_i``. Used by prepare to
    #: compute effective masses + bias.
    inv_mass: wp.float32
    #: World-frame inverse inertia. Zero matrix for triangle sides.
    inv_inertia: wp.mat33f


@wp.func
def side_pose_zero() -> SidePose:
    """Zero-initialised :class:`SidePose` -- handy as a return type
    placeholder before the kind-specific branch fills it in."""
    p = SidePose()
    p.position = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    p.orientation = wp.quatf(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
    p.body_com = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    p.inv_mass = wp.float32(0.0)
    p.inv_inertia = wp.mat33f(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )
    return p


@wp.struct
class SideKinematics:
    """Side-uniform kinematics for one contact endpoint, as the iterate
    sees it.

    Carries everything :func:`_make_contact_iterate_at` needs to compute the
    contact-point velocity (``v_world``), the row's contribution to
    ``J M^-1 J^T`` (``inv_mass`` + ``r`` + ``inv_inertia``), and to
    scatter the impulse back to the underlying body / particles via
    :func:`apply_rigid_side` / :func:`apply_triangle_side`.

    Field semantics:

    * Rigid side: ``v_world = v + omega x r`` with ``r = contact_p -
      body_position``. ``inv_mass`` / ``inv_inertia`` come straight
      from :class:`BodyContainer`.
    * Triangle side: ``v_world = sum_i w_i * v_i``. ``inv_mass = sum_i
      w_i^2 * inv_m_i``, ``r = 0``, ``inv_inertia = 0``: the
      Galerkin-FEM projection collapses the angular contribution into
      the diagonal mass. See module docstring.
    """

    #: World-space position of the contact point on this side. Used by
    #: prepare for the gap residual; iterate doesn't read it (lever
    #: arm is recomputed inline from the body pose / barycentric
    #: weights).
    p_world: wp.vec3f
    #: World-space velocity at the contact point. Includes the
    #: angular contribution ``omega x r`` for rigid sides; pure
    #: weighted-sum of node velocities for triangle sides.
    v_world: wp.vec3f
    #: World-space angular velocity (rigid side only -- zero for
    #: triangle). Threaded through so iterate can reconstruct
    #: ``v_world`` after applying an impulse without re-fetching from
    #: the body store. Triangle iterates never use this field.
    w_world: wp.vec3f
    #: World-space lever arm ``r``. Zero for triangle sides.
    r: wp.vec3f
    #: Diagonal inverse mass of the side at the contact point.
    inv_mass: wp.float32
    #: World-frame inverse inertia. Zero matrix for triangle sides.
    inv_inertia: wp.mat33f


@wp.func
def side_kinematics_zero() -> SideKinematics:
    """Zero-initialised :class:`SideKinematics`."""
    s = SideKinematics()
    s.p_world = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    s.v_world = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    s.w_world = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    s.r = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    s.inv_mass = wp.float32(0.0)
    s.inv_inertia = wp.mat33f(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )
    return s


# ---------------------------------------------------------------------------
# Rigid side
# ---------------------------------------------------------------------------


@wp.func
def fetch_rigid_pose(bodies: BodyContainer, b: wp.int32) -> SidePose:
    """Read a rigid side's pose (used by prepare).

    Returns the body's world position / orientation / COM offset and
    its inverse mass + world-frame inverse inertia. ``b`` is a body
    index into :class:`BodyContainer`. ``b < 0`` (the Newton "no
    rigid body" sentinel for world-fixed shapes) returns a
    zero-mass identity-pose placeholder so reads / writes elsewhere
    in the solver don't index ``bodies[-1]`` -- see
    :func:`read_rigid_column_state` for the iterate-side companion.
    """
    p = SidePose()
    if b >= wp.int32(0):
        p.position = bodies.position[b]
        p.orientation = bodies.orientation[b]
        p.body_com = bodies.body_com[b]
        p.inv_mass = bodies.inverse_mass[b]
        p.inv_inertia = bodies.inverse_inertia_world[b]
    else:
        p.position = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        p.orientation = wp.quatf(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
        p.body_com = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        p.inv_mass = wp.float32(0.0)
        p.inv_inertia = wp.mat33f(
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
        )
    return p


@wp.func
def fetch_rigid_side(
    bodies: BodyContainer,
    b: wp.int32,
    contact_p_world: wp.vec3f,
) -> SideKinematics:
    """Read a rigid side's kinematics at the given contact point.

    Computes ``r = contact_p_world - bodies.position[b]`` and
    ``v_world = v + omega x r``; copies inv_mass + inv_inertia from
    the body container. Used by the iterate's per-contact loop after
    prepare has already published the contact frame + per-contact
    biases into :class:`ContactContainer`.
    """
    s = SideKinematics()
    s.p_world = contact_p_world
    s.r = contact_p_world - bodies.position[b]
    v = bodies.velocity[b]
    w = bodies.angular_velocity[b]
    s.v_world = v + wp.cross(w, s.r)
    s.w_world = w
    s.inv_mass = bodies.inverse_mass[b]
    s.inv_inertia = bodies.inverse_inertia_world[b]
    return s


@wp.struct
class RigidColumnState:
    """Cached body-side state hoisted out of the per-contact loop.

    The contact iterate reads pose + inertia + velocity once per
    column and updates ``v`` / ``w`` in registers across the
    per-contact sweep, scattering once at end-of-column. This
    struct packages all six fields so the read / write helpers
    can transparently handle the ``b == -1`` "no rigid body"
    sentinel by returning a zero-mass identity-pose state and
    skipping the writes.
    """

    orientation: wp.quatf
    body_com: wp.vec3f
    inv_mass: wp.float32
    inv_inertia: wp.mat33f
    velocity: wp.vec3f
    angular_velocity: wp.vec3f


@wp.func
def read_rigid_column_state(bodies: BodyContainer, b: wp.int32) -> RigidColumnState:
    """Read a body's per-column state, or return a zero-mass static
    placeholder when ``b < 0``.

    Newton encodes "world-fixed" shapes as ``shape_body == -1`` --
    the iterate then carries ``b == -1`` through the contact column.
    Without this guard, the per-column hoists would do
    ``bodies.velocity[-1]`` etc., which (a) reads garbage and (b)
    writes garbage *into the byte before the bodies array* at
    end-of-column scatter; the latter corrupts adjacent
    allocations and crashes the partitioner on cloth-vs-static
    scenes where multiple contacts hammer the same OOB address.
    Returning a zero-state placeholder + skipping the write
    (see :func:`write_rigid_column_velocity`) makes the rigid hot
    path treat the static side as inert: every impulse term goes
    to zero (``inv_mass = 0`` / ``inv_inertia = 0``) and no
    bookkeeping write fires.
    """
    s = RigidColumnState()
    if b >= wp.int32(0):
        s.orientation = bodies.orientation[b]
        s.body_com = bodies.body_com[b]
        s.inv_mass = bodies.inverse_mass[b]
        s.inv_inertia = bodies.inverse_inertia_world[b]
        s.velocity = bodies.velocity[b]
        s.angular_velocity = bodies.angular_velocity[b]
    else:
        s.orientation = wp.quatf(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
        s.body_com = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        s.inv_mass = wp.float32(0.0)
        s.inv_inertia = wp.mat33f(
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
            wp.float32(0.0),
        )
        s.velocity = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        s.angular_velocity = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    return s


@wp.func
def write_rigid_column_velocity(
    bodies: BodyContainer,
    b: wp.int32,
    velocity: wp.vec3f,
    angular_velocity: wp.vec3f,
):
    """Scatter the per-column register-cached ``(v, w)`` back to the
    body store. ``b < 0`` (no rigid body) is silently skipped."""
    if b >= wp.int32(0):
        bodies.velocity[b] = velocity
        bodies.angular_velocity[b] = angular_velocity


@wp.func
def apply_rigid_side(
    bodies: BodyContainer,
    b: wp.int32,
    imp: wp.vec3f,
    r: wp.vec3f,
    sign: wp.float32,
):
    """Scatter a contact impulse onto a rigid side.

    Updates ``bodies.velocity[b]`` and ``bodies.angular_velocity[b]``
    by ``sign * inv_mass * imp`` and ``sign * inv_inertia @ cross(r,
    imp)`` respectively. ``sign`` lets the iterate negate the impulse
    on side A while applying ``+imp`` on side B (the contact-pair
    Newton's-third-law convention). Used in the *un-batched* fallback
    path; the rigid-rigid hot path keeps using
    :func:`~newton._src.solvers.phoenx.helpers.math_helpers.apply_pair_velocity_impulse`
    so per-contact body-velocity scatter stays batched at end-of-
    column (lower memory traffic, identical math).
    """
    inv_mass = bodies.inverse_mass[b]
    inv_inertia = bodies.inverse_inertia_world[b]
    bodies.velocity[b] = bodies.velocity[b] + (sign * inv_mass) * imp
    bodies.angular_velocity[b] = bodies.angular_velocity[b] + sign * (inv_inertia @ wp.cross(r, imp))


@wp.func
def apply_rigid_side_batched(
    v: wp.vec3f,
    w: wp.vec3f,
    inv_mass: wp.float32,
    inv_inertia: wp.mat33f,
    r: wp.vec3f,
    imp: wp.vec3f,
    sign: wp.float32,
):
    """In-register version of :func:`apply_rigid_side`.

    Returns ``(v', w')`` -- the updated linear / angular velocity
    pair. Lets the iterate keep body velocity / angular velocity in
    thread-local registers across the per-contact loop and scatter
    once at end-of-column (one global write per body per pair, not
    per contact). Used by the hot rigid-rigid path; triangle sides go
    through :func:`apply_triangle_side` because each contact mutates
    three independent particle slots that can't be coalesced into a
    single body's worth of registers.
    """
    v_new = v + (sign * inv_mass) * imp
    w_new = w + sign * (inv_inertia @ wp.cross(r, imp))
    return v_new, w_new


# ---------------------------------------------------------------------------
# Triangle side
# ---------------------------------------------------------------------------


@wp.func
def triangle_anchor_world(
    store: BodyOrParticleStore,
    body_a: wp.int32,
    body_b: wp.int32,
    body_c: wp.int32,
    weights: wp.vec3f,
) -> wp.vec3f:
    """Barycentric-weighted world-space position of a triangle's contact
    point.

    ``world_p = w_a * x_a + w_b * x_b + w_c * x_c``. Reads each
    particle's position through the unified body-or-particle store
    accessor so triangle sides whose nodes happen to be rigid-body
    anchors (future hybrid case) keep working transparently.
    """
    x_a = get_position(store, body_a)
    x_b = get_position(store, body_b)
    x_c = get_position(store, body_c)
    return weights[0] * x_a + weights[1] * x_b + weights[2] * x_c


@wp.func
def fetch_triangle_pose(
    store: BodyOrParticleStore,
    body_a: wp.int32,
    body_b: wp.int32,
    body_c: wp.int32,
    weights: wp.vec3f,
) -> SidePose:
    """Read a triangle side's pose (used by prepare).

    The "pose" of a triangle endpoint is degenerate: orientation is
    identity, ``body_com`` is zero, ``position`` is the barycentric-
    weighted world-space point. ``inv_mass`` is the Galerkin
    projection ``sum_i w_i^2 * inv_m_i`` (see module docstring);
    ``inv_inertia`` is the zero matrix.

    Pinned nodes (``inv_m_i == 0``) drop out of the sum cleanly. A
    fully pinned triangle has ``inv_mass == 0`` and is treated like a
    static body by the prepare's effective-mass calls.
    """
    p = SidePose()
    p.position = triangle_anchor_world(store, body_a, body_b, body_c, weights)
    p.orientation = wp.quatf(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
    p.body_com = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    # Galerkin projection: mass-weighted with weight^2 because the
    # impulse scatter is ``delta_v_i = w_i * inv_m_i * J`` and the
    # resulting contact-point velocity change is ``sum_i w_i *
    # delta_v_i = J * sum_i w_i^2 * inv_m_i``.
    inv_m_a = wp.float32(0.0)
    inv_m_b = wp.float32(0.0)
    inv_m_c = wp.float32(0.0)
    if is_particle(store, body_a):
        inv_m_a = store.particles.inverse_mass[body_a - store.num_bodies]
    else:
        inv_m_a = store.bodies.inverse_mass[body_a]
    if is_particle(store, body_b):
        inv_m_b = store.particles.inverse_mass[body_b - store.num_bodies]
    else:
        inv_m_b = store.bodies.inverse_mass[body_b]
    if is_particle(store, body_c):
        inv_m_c = store.particles.inverse_mass[body_c - store.num_bodies]
    else:
        inv_m_c = store.bodies.inverse_mass[body_c]
    p.inv_mass = (
        weights[0] * weights[0] * inv_m_a + weights[1] * weights[1] * inv_m_b + weights[2] * weights[2] * inv_m_c
    )
    p.inv_inertia = wp.mat33f(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )
    return p


@wp.func
def fetch_triangle_side(
    store: BodyOrParticleStore,
    body_a: wp.int32,
    body_b: wp.int32,
    body_c: wp.int32,
    weights: wp.vec3f,
) -> SideKinematics:
    """Read a triangle side's kinematics at the barycentric contact
    point.

    ``v_world = sum_i w_i * v_i`` over the three nodes. ``r = 0`` and
    ``inv_inertia = 0`` -- the angular DoF doesn't exist for a
    barycentric point (see module docstring). ``inv_mass`` is the
    Galerkin projection ``sum_i w_i^2 * inv_m_i``.

    The store accessors handle pinned nodes (``inv_m_i = 0``) and the
    rigid-attached-cloth-node case (a triangle node that happens to
    sit on a rigid body) transparently.
    """
    s = SideKinematics()
    x_a = get_position(store, body_a)
    x_b = get_position(store, body_b)
    x_c = get_position(store, body_c)
    v_a = get_velocity(store, body_a)
    v_b = get_velocity(store, body_b)
    v_c = get_velocity(store, body_c)
    s.p_world = weights[0] * x_a + weights[1] * x_b + weights[2] * x_c
    s.v_world = weights[0] * v_a + weights[1] * v_b + weights[2] * v_c
    s.w_world = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    s.r = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    inv_m_a = wp.float32(0.0)
    inv_m_b = wp.float32(0.0)
    inv_m_c = wp.float32(0.0)
    if is_particle(store, body_a):
        inv_m_a = store.particles.inverse_mass[body_a - store.num_bodies]
    else:
        inv_m_a = store.bodies.inverse_mass[body_a]
    if is_particle(store, body_b):
        inv_m_b = store.particles.inverse_mass[body_b - store.num_bodies]
    else:
        inv_m_b = store.bodies.inverse_mass[body_b]
    if is_particle(store, body_c):
        inv_m_c = store.particles.inverse_mass[body_c - store.num_bodies]
    else:
        inv_m_c = store.bodies.inverse_mass[body_c]
    s.inv_mass = (
        weights[0] * weights[0] * inv_m_a + weights[1] * weights[1] * inv_m_b + weights[2] * weights[2] * inv_m_c
    )
    s.inv_inertia = wp.mat33f(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )
    return s


@wp.func
def apply_triangle_side(
    store: BodyOrParticleStore,
    body_a: wp.int32,
    body_b: wp.int32,
    body_c: wp.int32,
    weights: wp.vec3f,
    imp: wp.vec3f,
    sign: wp.float32,
):
    """Scatter a contact impulse onto the three particles of a triangle
    side.

    Per-node update: ``delta_v_i = sign * w_i * inv_m_i * imp`` for
    ``i in {a, b, c}``. Pinned nodes (``inv_m_i == 0``) are
    automatically no-ops because the multiplicative factor collapses
    to zero -- no branch needed.

    Goes through :func:`set_velocity` (which routes to the body or
    particle store via the unified-index threshold) so this helper
    keeps working in the rigid-attached-cloth-node case where one of
    the triangle's vertices is anchored to a rigid body. For pure
    cloth triangles (the common case) every write lands in
    :class:`ParticleContainer.velocity` directly.

    Unlike :func:`apply_rigid_side_batched`, the per-contact triangle
    scatter is not register-batched across the column loop because
    the three node indices vary per contact (different contacts on
    the same shape pair routinely involve different triangle
    sub-faces). Each contact does three particle-velocity reads +
    writes; for typical cloth scenes this still sits well below the
    rigid path's per-contact body cost because the particle store is
    much narrower.
    """
    v_a = get_velocity(store, body_a)
    v_b = get_velocity(store, body_b)
    v_c = get_velocity(store, body_c)
    inv_m_a = wp.float32(0.0)
    inv_m_b = wp.float32(0.0)
    inv_m_c = wp.float32(0.0)
    if is_particle(store, body_a):
        inv_m_a = store.particles.inverse_mass[body_a - store.num_bodies]
    else:
        inv_m_a = store.bodies.inverse_mass[body_a]
    if is_particle(store, body_b):
        inv_m_b = store.particles.inverse_mass[body_b - store.num_bodies]
    else:
        inv_m_b = store.bodies.inverse_mass[body_b]
    if is_particle(store, body_c):
        inv_m_c = store.particles.inverse_mass[body_c - store.num_bodies]
    else:
        inv_m_c = store.bodies.inverse_mass[body_c]
    set_velocity(store, body_a, v_a + (sign * weights[0] * inv_m_a) * imp)
    set_velocity(store, body_b, v_b + (sign * weights[1] * inv_m_b) * imp)
    set_velocity(store, body_c, v_c + (sign * weights[2] * inv_m_c) * imp)
