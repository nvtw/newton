# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-endpoint state plumbing for the rigid contact constraint.

The PhoenX contact iterate / prepare runs the same PGS lambda math at
each endpoint. Body-frame anchor reconstruction, lever-arm cross-
products, mass-matrix assembly, and impulse application are factored
behind two ``wp.func`` accessors:

* :func:`endpoint_load` -- world contact point + per-endpoint
  velocity / inverse-mass / inverse-inertia / lever arm.
* :func:`endpoint_apply_impulse` -- distribute a 3-vector impulse to
  the endpoint's velocity store (linear via inverse mass; angular via
  inverse inertia and lever arm).

The contact column carries each endpoint as a body index ``b``;
the per-contact ``local_p`` slot in :class:`ContactContainer`
carries the body-local anchor (origin frame, see
:func:`contact_prepare_for_iteration_at`).
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer

__all__ = [
    "EndpointState",
    "endpoint_apply_impulse",
    "endpoint_load",
    "endpoint_velocity_at",
    "endpoint_warmstart_apply_impulse",
    "endpoint_world_point",
]


@wp.struct
class EndpointState:
    """Per-endpoint state at one contact point.

    Caller-local; not stored. :func:`endpoint_load` produces it once
    per (contact, endpoint) and the iterate / prepare lambda math
    consumes it for the rest of the contact iteration.
    """

    #: World-space position of the contact point on the endpoint.
    p_world: wp.vec3f
    #: Linear velocity at the contact point in world space.
    v: wp.vec3f
    #: Angular velocity.
    w: wp.vec3f
    #: Lever arm from the body's COM (== ``bodies.position``) to the
    #: contact point.
    r: wp.vec3f
    #: Inverse mass at the contact point.
    inv_mass: wp.float32
    #: Inverse inertia tensor (world frame).
    inv_inertia: wp.mat33f


# ---------------------------------------------------------------------------
# Endpoint load / apply.
# ---------------------------------------------------------------------------


@wp.func
def endpoint_load(
    idx: wp.int32,
    local_anchor: wp.vec3f,
    margin: wp.float32,
    margin_sign: wp.float32,
    n: wp.vec3f,
    bodies: BodyContainer,
) -> EndpointState:
    """Build the per-contact state for one rigid endpoint.

    Args:
        idx: Body index.
        local_anchor: Body-local anchor (origin frame).
        margin: Per-shape surface thickness ``[m]``.
        margin_sign: ``+1`` for endpoint 1 (push along ``+n``),
            ``-1`` for endpoint 2 (push along ``-n``). Mirrors the
            sign convention in :func:`contact_iterate_at`.
        n: World-space contact normal (pointing endpoint 1 ->
            endpoint 2). Used to apply the surface-margin shift.
        bodies: Rigid-body SoA (read).
    """
    out = EndpointState()
    position = bodies.position[idx]
    orientation = bodies.orientation[idx]
    body_com = bodies.body_com[idx]
    out.p_world = position + wp.quat_rotate(orientation, local_anchor - body_com) + margin_sign * margin * n
    out.v = bodies.velocity[idx]
    out.w = bodies.angular_velocity[idx]
    out.r = out.p_world - position
    out.inv_mass = bodies.inverse_mass[idx]
    out.inv_inertia = bodies.inverse_inertia_world[idx]
    return out


@wp.func
def endpoint_apply_impulse(
    idx: wp.int32,
    imp: wp.vec3f,
    inv_mass: wp.float32,
    inv_inertia: wp.mat33f,
    r: wp.vec3f,
    sign: wp.float32,
    bodies: BodyContainer,
):
    """Apply a 3-vector world-space impulse at the contact point.

    Writes directly to the body's linear and angular velocity stores.
    Graph coloring guarantees no two same-color contacts share an
    endpoint, so plain stores are race-free.
    """
    bodies.velocity[idx] = bodies.velocity[idx] + sign * inv_mass * imp
    bodies.angular_velocity[idx] = bodies.angular_velocity[idx] + sign * inv_inertia @ wp.cross(r, imp)


@wp.func
def endpoint_warmstart_apply_impulse(
    idx: wp.int32,
    lin_imp: wp.vec3f,
    ang_imp: wp.vec3f,
    inv_mass: wp.float32,
    inv_inertia: wp.mat33f,
    sign: wp.float32,
    bodies: BodyContainer,
):
    """Scatter the batched warm-start impulse onto the body's velocity store.

    Mirrors the prepare's tail scatter
    (``bodies.velocity[b] += sign * inv_m * lin_imp``,
    ``bodies.angular_velocity[b] += sign * inv_I @ ang_imp``). Used by
    :func:`contact_prepare_for_iteration_at` after the per-contact loop
    has accumulated ``lin_imp`` / ``ang_imp``.
    """
    bodies.velocity[idx] = bodies.velocity[idx] + sign * inv_mass * lin_imp
    bodies.angular_velocity[idx] = bodies.angular_velocity[idx] + sign * inv_inertia @ ang_imp


# ---------------------------------------------------------------------------
# Anchor-only helpers (no margin shift, no full state -- used by the
# warm-start gather to compare prev / fresh anchor positions and the
# contact-relative point velocity).
# ---------------------------------------------------------------------------


@wp.func
def endpoint_world_point(
    idx: wp.int32,
    local_anchor: wp.vec3f,
    bodies: BodyContainer,
) -> wp.vec3f:
    """World-space anchor point for one rigid endpoint.

    ``local_anchor`` is the body-local origin-frame point; the
    COM-corrected world transform applies. No margin shift is applied
    -- callers comparing anchors across frames want the un-shifted point
    so the comparison is independent of per-shape ``shape_gap``.
    """
    position = bodies.position[idx]
    orientation = bodies.orientation[idx]
    body_com = bodies.body_com[idx]
    return position + wp.quat_rotate(orientation, local_anchor - body_com)


@wp.func
def endpoint_velocity_at(
    idx: wp.int32,
    local_anchor: wp.vec3f,
    bodies: BodyContainer,
) -> wp.vec3f:
    """World linear velocity at the rigid body's anchor point.

    ``v + omega x r`` where ``r`` is the lever arm from the COM to the
    anchor. Used by the warm-start gather to seed the friction tangent
    from the contact-relative slip velocity.

    ``r`` is computed via ``quat_rotate`` directly (no subtraction
    round-trip through world space) so the value is bit-equivalent to
    the inline expression the rigid contact ingest used previously --
    a subtle ``(p + r) - p`` cancellation otherwise shifts the LSBs
    enough to perturb settled-rest convergence in long contact stacks
    (~10 % regression on the 5-layer pyramid).
    """
    orientation = bodies.orientation[idx]
    body_com = bodies.body_com[idx]
    r = wp.quat_rotate(orientation, local_anchor - body_com)
    return bodies.velocity[idx] + wp.cross(bodies.angular_velocity[idx], r)
