# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-endpoint state plumbing for the unified contact constraint.

The PhoenX contact iterate / prepare runs the same PGS lambda math
regardless of whether each endpoint is a rigid body or a cloth
triangle. Body-frame anchor reconstruction, lever-arm cross-products,
mass-matrix assembly, and impulse application differ; the friction
cone, soft-PD branch, and lambda updates do not.

This module isolates the differences behind two ``wp.func`` accessors
driven by a per-endpoint kind tag stored in each contact column:

* :func:`endpoint_load` -- world contact point + per-endpoint
  velocity / inverse-mass / inverse-inertia / lever arm.
* :func:`endpoint_apply_impulse` -- distribute a 3-vector impulse to
  the endpoint's velocity store(s) (rigid: COM linear + angular;
  triangle: per-vertex linear by barycentric weight).

For the triangle endpoint, the impulse distributes via barycentric
weights:

.. code-block:: text

    for i in (a, b, c):
        particles.velocity[p_i] += w_i * inv_m_i * imp

There is no angular term, so :func:`effective_mass_scalar`'s
``dot(rc, I^{-1} rc)`` term vanishes when the endpoint is fed
``inv_inertia = 0``. The resulting pre-folded scalar mass is the
mass-weighted sum ``sum(w_i^2 * inv_m_i)`` -- the standard Macklin
RT/TT point-mass reduction.

The contact column carries each endpoint as a ``(kind, idx)`` pair:

* :data:`ENDPOINT_KIND_RIGID` -- ``idx`` is a rigid body index;
  the per-contact ``local_p`` slot in :class:`ContactContainer`
  carries the body-local anchor (origin frame, see
  :func:`contact_prepare_for_iteration_at`).
* :data:`ENDPOINT_KIND_TRIANGLE` -- ``idx`` is a triangle index
  ``t in [0, T)``; the per-contact ``local_p`` slot carries
  barycentric weights ``(w_a, w_b, w_c)`` packed as a ``vec3f`` and
  the three particle indices come from
  ``solver.tri_indices[t]``.

## Cross-kind access-mode handling

Cloth (POSITION_LEVEL) and rigid-vs-triangle contact (VELOCITY_LEVEL)
share particle endpoints across alternating graph colours. Each
particle carries a per-entity ``access_mode`` flag (see
:mod:`newton._src.solvers.phoenx.access_mode`); the triangle branch
of :func:`endpoint_load` flips its three vertices to
``VELOCITY_LEVEL`` via :func:`particle_set_access_mode` before
reading ``particles.velocity``. The flip is the
:func:`~newton._src.solvers.phoenx.access_mode.synchronize_position_velocity`
finite-diff: any cloth-written position correction in
``particles.position`` is converted into the equivalent velocity
delta against ``position_prev_substep``. The contact iterate then
reads ``particles.velocity`` directly and writes velocity impulses
into it; if a subsequent cloth iterate touches the same particle it
flips back to ``POSITION_LEVEL`` and re-integrates ``position =
position_prev_substep + velocity * dt``, which preserves both the
cloth correction and the contact impulse. Direct port of Jitter2's
``TinyRigidState.SetAccessMode`` pattern
(``MassSplitting/TinyRigidState.cs``).
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_POSITION_LEVEL, ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.body_or_particle import particle_set_access_mode
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "ENDPOINT_KIND_RIGID",
    "ENDPOINT_KIND_TRIANGLE",
    "EndpointState",
    "endpoint_apply_impulse",
    "endpoint_load",
    "endpoint_velocity_at",
    "endpoint_warmstart_apply_impulse",
    "endpoint_world_point",
]


_ACCESS_MODE_VELOCITY_LEVEL = wp.constant(wp.int32(ACCESS_MODE_VELOCITY_LEVEL))
_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))


#: Endpoint kind tags. Stored as int32 dwords on each contact column.
ENDPOINT_KIND_RIGID = wp.constant(wp.int32(0))
ENDPOINT_KIND_TRIANGLE = wp.constant(wp.int32(1))


@wp.struct
class EndpointState:
    """Per-endpoint state at one contact point.

    Caller-local; not stored. :func:`endpoint_load` produces it once
    per (contact, endpoint) and the iterate / prepare lambda math
    consumes it for the rest of the contact iteration.

    Fields collapse to the rigid case verbatim when ``kind ==
    RIGID`` (``r`` is the lever arm to the COM, ``inv_inertia`` is
    the body's world inverse inertia tensor) and to the
    point-mass-on-a-triangle case when ``kind == TRIANGLE``
    (``r = 0``, ``w = 0``, ``inv_inertia = 0``, ``v`` and
    ``inv_mass`` are the barycentric reductions over the three
    vertices).
    """

    #: World-space position of the contact point on the endpoint.
    p_world: wp.vec3f
    #: Linear velocity at the contact point in world space.
    v: wp.vec3f
    #: Angular velocity (zero for triangle endpoints).
    w: wp.vec3f
    #: Lever arm from the endpoint's reference frame origin to the
    #: contact point. Rigid: ``p_world - bodies.position``. Triangle:
    #: zero (the contact point IS the reference point for the
    #: per-vertex impulse distribution).
    r: wp.vec3f
    #: Effective inverse mass at the contact point. Rigid:
    #: ``bodies.inverse_mass``. Triangle: ``sum(w_i^2 * inv_m_i)``.
    inv_mass: wp.float32
    #: Effective inverse inertia tensor (zero for triangle).
    inv_inertia: wp.mat33f


# ---------------------------------------------------------------------------
# Triangle helpers.
# ---------------------------------------------------------------------------


@wp.func
def _triangle_vertex_indices(
    tri_indices: wp.array[wp.vec4i],
    t: wp.int32,
) -> wp.vec3i:
    """Particle indices ``(p_a, p_b, p_c)`` of triangle ``t``."""
    idx4 = tri_indices[t]
    return wp.vec3i(idx4[0], idx4[1], idx4[2])


@wp.func
def _triangle_world_point(
    particles: ParticleContainer,
    pa: wp.int32,
    pb: wp.int32,
    pc: wp.int32,
    bary: wp.vec3f,
) -> wp.vec3f:
    """World position of a barycentric point on a triangle."""
    return bary[0] * particles.position[pa] + bary[1] * particles.position[pb] + bary[2] * particles.position[pc]


@wp.func
def _triangle_point_velocity(
    particles: ParticleContainer,
    pa: wp.int32,
    pb: wp.int32,
    pc: wp.int32,
    bary: wp.vec3f,
) -> wp.vec3f:
    """World linear velocity at a barycentric point on a triangle.

    Reads :attr:`ParticleContainer.velocity` directly. Use this only
    where the per-vertex ``access_mode`` is known to be
    ``VELOCITY_LEVEL`` (e.g., the warm-start gather at substep
    start). Inside the iterate, callers must first sync each vertex
    via :func:`_triangle_sync_to_velocity_level` so cloth's
    position-level writes are folded into ``velocity``.
    """
    return bary[0] * particles.velocity[pa] + bary[1] * particles.velocity[pb] + bary[2] * particles.velocity[pc]


@wp.func
def _triangle_sync_to_velocity_level(
    particles: ParticleContainer,
    pa: wp.int32,
    pb: wp.int32,
    pc: wp.int32,
    idt: wp.float32,
):
    """Flip the three triangle vertices' access mode to ``VELOCITY_LEVEL``.

    Direct call to
    :func:`~newton._src.solvers.phoenx.body_or_particle.particle_set_access_mode`
    on each vertex. For a vertex already in ``VELOCITY_LEVEL`` (or
    ``STATIC``), the synchronize helper short-circuits; for a vertex
    in ``POSITION_LEVEL`` (cloth has just written its position) the
    helper runs the
    :func:`~newton._src.solvers.phoenx.access_mode.synchronize_position_velocity`
    finite-diff against ``position_prev_substep`` so the subsequent
    ``particles.velocity[p]`` read picks up the cloth-corrected
    velocity. Mirrors Jitter2's
    ``TinyRigidState.SetAccessMode(VelocityLevel)``
    (``MassSplitting/TinyRigidState.cs:108``).
    """
    particle_set_access_mode(particles, pa, _ACCESS_MODE_VELOCITY_LEVEL, idt)
    particle_set_access_mode(particles, pb, _ACCESS_MODE_VELOCITY_LEVEL, idt)
    particle_set_access_mode(particles, pc, _ACCESS_MODE_VELOCITY_LEVEL, idt)


@wp.func
def _triangle_point_inv_mass(
    particles: ParticleContainer,
    pa: wp.int32,
    pb: wp.int32,
    pc: wp.int32,
    bary: wp.vec3f,
) -> wp.float32:
    """Effective inverse mass at a barycentric point on a triangle.

    Standard Macklin point-on-triangle reduction:
    ``inv_m_eff = sum_i w_i^2 * inv_m_i``. With this scalar fed to
    :func:`effective_mass_scalar` (alongside ``inv_inertia = 0``),
    the rigid contact lambda math is bit-exact for the triangle
    endpoint.
    """
    return (
        bary[0] * bary[0] * particles.inverse_mass[pa]
        + bary[1] * bary[1] * particles.inverse_mass[pb]
        + bary[2] * bary[2] * particles.inverse_mass[pc]
    )


@wp.func
def _triangle_rigid_body_state(
    particles: ParticleContainer,
    pa: wp.int32,
    pb: wp.int32,
    pc: wp.int32,
):
    """Treat the three triangle vertices as a virtual rigid body and
    return its mass, centroid, inertia, lever arms, linear velocity,
    and angular velocity.

    The contact iteration uses this to compute the contact effective
    mass (linear + rotational impedance) and to redistribute the
    contact impulse onto the three particles such that BOTH linear
    and angular momentum about the centroid are preserved (which is
    the property the user's reference rigid-cloth example gets for
    free because each tri there *is* an actual rigid body).

    Pinned particles (``inverse_mass == 0``) are excluded entirely:
    their position and velocity remain fixed by the access-mode
    STATIC short-circuit, so their contribution to the centroid /
    inertia / angular momentum is ill-defined (infinite mass).  We
    compute the centroid / inertia from the *free* particles only;
    the rigid-body update is then applied only to free particles.

    Returns ``(centroid, total_mass, m_a, m_b, m_c, inv_inertia,
    omega, v_centroid, normal)``.  The triangle normal ``normal``
    is returned so callers can project rotation updates out of the
    null-space axis (3 coplanar mass points have rank-2 inertia).
    """
    inv_m_a = particles.inverse_mass[pa]
    inv_m_b = particles.inverse_mass[pb]
    inv_m_c = particles.inverse_mass[pc]
    m_a = wp.float32(0.0)
    m_b = wp.float32(0.0)
    m_c = wp.float32(0.0)
    if inv_m_a > wp.float32(0.0):
        m_a = wp.float32(1.0) / inv_m_a
    if inv_m_b > wp.float32(0.0):
        m_b = wp.float32(1.0) / inv_m_b
    if inv_m_c > wp.float32(0.0):
        m_c = wp.float32(1.0) / inv_m_c
    total_mass = m_a + m_b + m_c

    pos_a = particles.position[pa]
    pos_b = particles.position[pb]
    pos_c = particles.position[pc]
    vel_a = particles.velocity[pa]
    vel_b = particles.velocity[pb]
    vel_c = particles.velocity[pc]

    # Triangle normal -- always derivable from the geometric corners,
    # even if some are pinned.  Used to regularise the rank-deficient
    # inertia tensor along the null-space axis.
    e1 = pos_b - pos_a
    e2 = pos_c - pos_a
    n_unnorm = wp.cross(e1, e2)
    n_len_sq = wp.dot(n_unnorm, n_unnorm)
    normal = wp.vec3f(0.0, 0.0, 1.0)
    if n_len_sq > wp.float32(1.0e-20):
        normal = n_unnorm / wp.sqrt(n_len_sq)

    if total_mass <= wp.float32(0.0):
        # Fully pinned triangle -- treat as infinite mass so any
        # contact impulse it sees produces zero motion.  The caller
        # uses ``inv_mass = 0`` and ``inv_inertia = 0`` for this case
        # (matching the rigid pinned-body convention).
        return (
            pos_a,
            wp.float32(0.0),
            wp.float32(0.0), wp.float32(0.0), wp.float32(0.0),
            wp.mat33f(0.0),
            wp.vec3f(0.0, 0.0, 0.0),
            wp.vec3f(0.0, 0.0, 0.0),
            normal,
        )

    inv_total = wp.float32(1.0) / total_mass
    centroid = (m_a * pos_a + m_b * pos_b + m_c * pos_c) * inv_total
    v_centroid = (m_a * vel_a + m_b * vel_b + m_c * vel_c) * inv_total

    r_a = pos_a - centroid
    r_b = pos_b - centroid
    r_c = pos_c - centroid

    # Inertia tensor about centroid (sum of point-mass inertias).
    rr_a = wp.dot(r_a, r_a)
    rr_b = wp.dot(r_b, r_b)
    rr_c = wp.dot(r_c, r_c)
    i00 = m_a * (rr_a - r_a[0] * r_a[0]) + m_b * (rr_b - r_b[0] * r_b[0]) + m_c * (rr_c - r_c[0] * r_c[0])
    i11 = m_a * (rr_a - r_a[1] * r_a[1]) + m_b * (rr_b - r_b[1] * r_b[1]) + m_c * (rr_c - r_c[1] * r_c[1])
    i22 = m_a * (rr_a - r_a[2] * r_a[2]) + m_b * (rr_b - r_b[2] * r_b[2]) + m_c * (rr_c - r_c[2] * r_c[2])
    i01 = -m_a * r_a[0] * r_a[1] - m_b * r_b[0] * r_b[1] - m_c * r_c[0] * r_c[1]
    i02 = -m_a * r_a[0] * r_a[2] - m_b * r_b[0] * r_b[2] - m_c * r_c[0] * r_c[2]
    i12 = -m_a * r_a[1] * r_a[2] - m_b * r_b[1] * r_b[2] - m_c * r_c[1] * r_c[2]

    # 3 non-collinear coplanar mass points have FULL-RANK inertia:
    # by the perpendicular-axis theorem, ``I_zz = I_xx + I_yy``
    # (where z is the triangle normal), so the tensor is rank 3
    # unless the triangle is degenerate.
    trace = i00 + i11 + i22
    reg = wp.max(wp.float32(1.0e-18), trace * wp.float32(1.0e-9))
    inertia_reg = wp.mat33f(
        i00 + reg, i01, i02,
        i01, i11 + reg, i12,
        i02, i12, i22 + reg,
    )
    inv_inertia = wp.inverse(inertia_reg)

    L = m_a * wp.cross(r_a, vel_a) + m_b * wp.cross(r_b, vel_b) + m_c * wp.cross(r_c, vel_c)
    omega = inv_inertia @ L

    return centroid, total_mass, m_a, m_b, m_c, inv_inertia, omega, v_centroid, normal


# ---------------------------------------------------------------------------
# Endpoint load / apply.
# ---------------------------------------------------------------------------


@wp.func
def endpoint_load(
    kind: wp.int32,
    idx: wp.int32,
    local_anchor: wp.vec3f,
    margin: wp.float32,
    margin_sign: wp.float32,
    n: wp.vec3f,
    idt: wp.float32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    tri_indices: wp.array[wp.vec4i],
) -> EndpointState:
    """Build the per-contact state for one endpoint.

    Iterate-time loader: triangle endpoints first sync their three
    vertices to ``VELOCITY_LEVEL`` via
    :func:`_triangle_sync_to_velocity_level`, then read
    ``particles.velocity`` directly. Cloth's most recent position
    correction is folded into the velocity field by the access-mode
    sync, so contact reads stay consistent across alternating cloth
    / contact colours within one PGS sweep.

    Args:
        kind: :data:`ENDPOINT_KIND_RIGID` or
            :data:`ENDPOINT_KIND_TRIANGLE`.
        idx: For RIGID, body index; for TRIANGLE, triangle index.
        local_anchor: For RIGID, body-local anchor (origin frame).
            For TRIANGLE, barycentric weights ``(w_a, w_b, w_c)``.
        margin: Per-shape surface thickness ``[m]``.
        margin_sign: ``+1`` for endpoint 1 (push along ``+n``),
            ``-1`` for endpoint 2 (push along ``-n``). Mirrors the
            sign convention in :func:`contact_iterate_at`.
        n: World-space contact normal (pointing endpoint 1 ->
            endpoint 2). Used to apply the surface-margin shift.
        idt: Inverse substep dt ``[1/s]``. Drives the per-vertex
            access-mode sync (Position-level -> Velocity-level
            finite-diff against ``position_prev_substep``).
        bodies: Rigid-body SoA (read).
        particles: Particle SoA (read; triangle endpoints only).
        tri_indices: Per-triangle ``vec4i`` particle indices
            ``(p_a, p_b, p_c, -1)``.
    """
    out = EndpointState()
    if kind == ENDPOINT_KIND_RIGID:
        position = bodies.position[idx]
        orientation = bodies.orientation[idx]
        body_com = bodies.body_com[idx]
        # Body-local origin-frame anchor -> world; offset along the
        # contact normal by the per-shape margin (matches the rigid
        # prepare convention).
        out.p_world = position + wp.quat_rotate(orientation, local_anchor - body_com) + margin_sign * margin * n
        out.v = bodies.velocity[idx]
        out.w = bodies.angular_velocity[idx]
        out.r = out.p_world - position
        out.inv_mass = bodies.inverse_mass[idx]
        out.inv_inertia = bodies.inverse_inertia_world[idx]
        return out

    # Triangle endpoint -- barycentric load + barycentric distribute
    # (the original, well-tested code path).  The "rigid body during
    # contact" experiment (committed and reverted: see commit history)
    # showed that pure rigid-body redistribution dramatically reduces
    # cube xy drift (0.09 m -> 0.001 m) but blows up after ~15 frames
    # because adjacent triangles sharing a particle each try to
    # enforce their own rigid-body rotation on it, and the iteration
    # amplifies through the shared vertex.  Until that shared-vertex
    # issue is resolved, we keep the bary path which is stable but
    # accepts higher xy drift.
    verts = _triangle_vertex_indices(tri_indices, idx)
    pa = verts[0]
    pb = verts[1]
    pc = verts[2]
    _triangle_sync_to_velocity_level(particles, pa, pb, pc, idt)
    out.p_world = _triangle_world_point(particles, pa, pb, pc, local_anchor) + margin_sign * margin * n
    out.v = _triangle_point_velocity(particles, pa, pb, pc, local_anchor)
    out.w = wp.vec3f(0.0, 0.0, 0.0)
    out.r = wp.vec3f(0.0, 0.0, 0.0)
    out.inv_mass = _triangle_point_inv_mass(particles, pa, pb, pc, local_anchor)
    out.inv_inertia = wp.mat33f(0.0)
    return out


@wp.func
def endpoint_apply_impulse(
    kind: wp.int32,
    idx: wp.int32,
    bary: wp.vec3f,
    imp: wp.vec3f,
    inv_mass: wp.float32,
    inv_inertia: wp.mat33f,
    r: wp.vec3f,
    sign: wp.float32,
    dt: wp.float32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    tri_indices: wp.array[wp.vec4i],
):
    """Apply a 3-vector world-space impulse at the contact point.

    Writes directly to the endpoint's velocity store. Graph coloring
    guarantees no two same-color contacts share an endpoint, so plain
    stores are race-free.

    Args:
        kind / idx: Endpoint discriminator.
        bary: Barycentric weights (TRIANGLE); ignored for RIGID.
        imp: World-space impulse [N*s].
        inv_mass / inv_inertia / r: Endpoint mass-matrix entries.
        sign: ``+1`` for endpoint 2, ``-1`` for endpoint 1.
        dt: Substep dt ``[s]``. Used to drive the per-vertex
            ``VELOCITY_LEVEL -> POSITION_LEVEL`` flip after the velocity
            write so cloth's substep-exit XPBD recovery preserves the
            contact impulse via the access-mode finite-diff.
    """
    if kind == ENDPOINT_KIND_RIGID:
        bodies.velocity[idx] = bodies.velocity[idx] + sign * inv_mass * imp
        bodies.angular_velocity[idx] = bodies.angular_velocity[idx] + sign * inv_inertia @ wp.cross(r, imp)
        return

    verts = _triangle_vertex_indices(tri_indices, idx)
    pa = verts[0]
    pb = verts[1]
    pc = verts[2]
    # ``r`` and ``inv_inertia`` were computed by
    # :func:`endpoint_load` against the virtual rigid-body view of
    # the triangle, so the lambda math sees the correct rotational
    # impedance.  Distribute the resulting impulse via the standard
    # barycentric weights -- conserves linear momentum AND angular
    # momentum about the (mass-weighted) centroid, while sidestepping
    # the shared-vertex amplification that a true rigid-body
    # redistribution suffers when adjacent triangles each try to
    # enforce their own rotation on the same particle.
    inv_m_a = particles.inverse_mass[pa]
    inv_m_b = particles.inverse_mass[pb]
    inv_m_c = particles.inverse_mass[pc]
    particles.velocity[pa] = particles.velocity[pa] + sign * bary[0] * inv_m_a * imp
    particles.velocity[pb] = particles.velocity[pb] + sign * bary[1] * inv_m_b * imp
    particles.velocity[pc] = particles.velocity[pc] + sign * bary[2] * inv_m_c * imp

    inv_dt = wp.float32(1.0) / dt
    particle_set_access_mode(particles, pa, _ACCESS_MODE_POSITION_LEVEL, inv_dt)
    particle_set_access_mode(particles, pb, _ACCESS_MODE_POSITION_LEVEL, inv_dt)
    particle_set_access_mode(particles, pc, _ACCESS_MODE_POSITION_LEVEL, inv_dt)


@wp.func
def endpoint_warmstart_apply_impulse(
    kind: wp.int32,
    idx: wp.int32,
    bary: wp.vec3f,
    lin_imp: wp.vec3f,
    ang_imp: wp.vec3f,
    inv_mass: wp.float32,
    inv_inertia: wp.mat33f,
    sign: wp.float32,
    dt: wp.float32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    tri_indices: wp.array[wp.vec4i],
):
    """Scatter the batched warm-start impulse onto the endpoint stores.

    Mirrors the prepare's tail scatter
    (``bodies.velocity[b] += sign * inv_m * lin_imp``,
    ``bodies.angular_velocity[b] += sign * inv_I @ ang_imp``) for
    rigid endpoints and the per-vertex scatter for triangle ones.
    Used by :func:`contact_prepare_for_iteration_at` after the
    per-contact loop has accumulated ``lin_imp`` / ``ang_imp``.

    The angular impulse is ignored for triangle endpoints (no
    rotational DOF). The linear impulse goes through the same
    barycentric per-vertex distribution as the iterate path, plus the
    matching ``VELOCITY_LEVEL -> POSITION_LEVEL`` access-mode flip so
    cloth's substep-exit recovery folds the warm-start impulse into
    ``particles.position`` via the finite-diff identity.
    """
    if kind == ENDPOINT_KIND_RIGID:
        bodies.velocity[idx] = bodies.velocity[idx] + sign * inv_mass * lin_imp
        bodies.angular_velocity[idx] = bodies.angular_velocity[idx] + sign * inv_inertia @ ang_imp
        return

    verts = _triangle_vertex_indices(tri_indices, idx)
    pa = verts[0]
    pb = verts[1]
    pc = verts[2]
    inv_m_a = particles.inverse_mass[pa]
    inv_m_b = particles.inverse_mass[pb]
    inv_m_c = particles.inverse_mass[pc]
    particles.velocity[pa] = particles.velocity[pa] + sign * bary[0] * inv_m_a * lin_imp
    particles.velocity[pb] = particles.velocity[pb] + sign * bary[1] * inv_m_b * lin_imp
    particles.velocity[pc] = particles.velocity[pc] + sign * bary[2] * inv_m_c * lin_imp

    inv_dt = wp.float32(1.0) / dt
    particle_set_access_mode(particles, pa, _ACCESS_MODE_POSITION_LEVEL, inv_dt)
    particle_set_access_mode(particles, pb, _ACCESS_MODE_POSITION_LEVEL, inv_dt)
    particle_set_access_mode(particles, pc, _ACCESS_MODE_POSITION_LEVEL, inv_dt)


# ---------------------------------------------------------------------------
# Anchor-only helpers (no margin shift, no full state -- used by the
# warm-start gather to compare prev / fresh anchor positions and the
# contact-relative point velocity).
# ---------------------------------------------------------------------------


@wp.func
def endpoint_world_point(
    kind: wp.int32,
    idx: wp.int32,
    local_anchor: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    tri_indices: wp.array[wp.vec4i],
) -> wp.vec3f:
    """World-space anchor point for one endpoint.

    For RIGID, ``local_anchor`` is the body-local origin-frame point;
    the COM-corrected world transform applies. For TRIANGLE,
    ``local_anchor`` is barycentric weights and the world point is the
    weighted sum over the three particle positions.

    No margin shift is applied -- callers comparing anchors across
    frames want the un-shifted point so the comparison is independent
    of per-shape ``shape_gap``.
    """
    if kind == ENDPOINT_KIND_RIGID:
        position = bodies.position[idx]
        orientation = bodies.orientation[idx]
        body_com = bodies.body_com[idx]
        return position + wp.quat_rotate(orientation, local_anchor - body_com)

    verts = _triangle_vertex_indices(tri_indices, idx)
    return _triangle_world_point(particles, verts[0], verts[1], verts[2], local_anchor)


@wp.func
def endpoint_velocity_at(
    kind: wp.int32,
    idx: wp.int32,
    local_anchor: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    tri_indices: wp.array[wp.vec4i],
) -> wp.vec3f:
    """World linear velocity at the endpoint's anchor point.

    Rigid: ``v + omega x r`` where ``r`` is the lever arm from the COM
    to the anchor. Triangle: barycentric reduction of vertex velocities.
    Used by the warm-start gather to seed the friction tangent from the
    contact-relative slip velocity.

    ``r`` is computed via ``quat_rotate`` directly (no subtraction
    round-trip through world space) so the value is bit-equivalent to
    the inline expression the rigid contact ingest used previously --
    a subtle ``(p + r) - p`` cancellation otherwise shifts the LSBs
    enough to perturb settled-rest convergence in long contact stacks
    (~10 % regression on the 5-layer pyramid).
    """
    if kind == ENDPOINT_KIND_RIGID:
        orientation = bodies.orientation[idx]
        body_com = bodies.body_com[idx]
        r = wp.quat_rotate(orientation, local_anchor - body_com)
        return bodies.velocity[idx] + wp.cross(bodies.angular_velocity[idx], r)

    verts = _triangle_vertex_indices(tri_indices, idx)
    return _triangle_point_velocity(particles, verts[0], verts[1], verts[2], local_anchor)
