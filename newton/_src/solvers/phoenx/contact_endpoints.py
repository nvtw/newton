# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact endpoint metadata for rigid and deformable PGS rows."""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_set_body1,
    contact_set_body2,
    contact_set_side0_kind,
    contact_set_side0_nodes_extra,
    contact_set_side1_kind,
    contact_set_side1_nodes_extra,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_set_side0_bary,
    cc_set_side1_bary,
)
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE",
    "SHAPE_ENDPOINT_KIND_RIGID",
    "SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON",
    "ShapeEndpoint",
    "pack_contact_barycentric_kernel",
    "pack_contact_endpoints_kernel",
    "populate_shape_endpoints_kernel",
    "shape_endpoints_zeros",
]


# ---------------------------------------------------------------------------
# Per-shape endpoint table
# ---------------------------------------------------------------------------
#
# Every shape in the unified shape array (rigid prefix [0, S), cloth-tri
# suffix [S, S+T)) carries an endpoint descriptor that the contact ingest
# kernel uses to translate ``(shape_a, shape_b)`` pairs into:
#
# * a primary unified body-or-particle node index per side (= ``body1`` /
#   ``body2`` of the contact column),
# * up to two extra unified-index particle nodes per cloth side
#   (= ``side*_nodes_extra``), and
# * a kind tag (rigid / cloth) so the iterate's endpoint helper knows
#   which container (BodyContainer / ParticleContainer) to read.
#
# The descriptor is populated once at scene build via
# the deformable contact setup and read by the
# contact-ingest kernel as a single 16-byte load per shape.

#: Shape is owned by a rigid body. ``nodes`` holds ``(body_unified, -1, -1, -1)``.
SHAPE_ENDPOINT_KIND_RIGID: int = 0

#: Shape is a cloth triangle. ``nodes`` holds three particle indices in
#: unified body-or-particle space ``(num_bodies + p_a, num_bodies + p_b,
#: num_bodies + p_c, -1)`` -- the 4th slot is unused for triangles.
SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE: int = 1

#: Shape is a soft-body tetrahedron. ``nodes`` holds four particle
#: indices in unified body-or-particle space ``(num_bodies + p_a,
#: num_bodies + p_b, num_bodies + p_c, num_bodies + p_d)``.
SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON: int = 2


@wp.struct
class ShapeEndpoint:
    """Per-shape mapping ``shape_index -> (kind, nodes)``.

    For rigid shapes, ``nodes[0]`` is the rigid body index in unified
    body-or-particle space (``[0, num_bodies)``); ``nodes[1..3]`` are
    ``-1``.

    For cloth-triangle shapes, ``nodes[0/1/2]`` are the three triangle
    particle indices in unified space (``num_bodies + particle_id``);
    ``nodes[3]`` is ``-1``.

    For soft-tet shapes, ``nodes[0..3]`` are the four tetrahedron vertex
    indices in unified space.
    """

    nodes: wp.vec4i
    kind: wp.int32


def shape_endpoints_zeros(num_shapes: int, device=None) -> wp.array[ShapeEndpoint]:
    """Allocate a zero-initialised :class:`ShapeEndpoint` array of
    length ``num_shapes``."""
    return wp.zeros(int(num_shapes), dtype=ShapeEndpoint, device=device)


@wp.kernel(enable_backward=False)
def populate_shape_endpoints_kernel(
    shape_body: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    cloth_shape_offset: wp.int32,
    num_cloth_triangles: wp.int32,
    soft_tet_shape_offset: wp.int32,
    num_soft_tetrahedra: wp.int32,
    num_bodies: wp.int32,
    phoenx_body_offset: wp.int32,
    # out
    shape_endpoints: wp.array[ShapeEndpoint],
):
    """One thread per shape ``s``: stamp its :class:`ShapeEndpoint`.

    Layout in the unified shape array:

    * ``s < cloth_shape_offset`` -> rigid: ``nodes = (newton_body +
      phoenx_body_offset, -1, -1, -1)``.
    * ``cloth_shape_offset <= s < soft_tet_shape_offset`` -> cloth tri:
      nodes = ``(num_bodies + p_a, num_bodies + p_b, num_bodies + p_c, -1)``.
    * ``s >= soft_tet_shape_offset`` -> soft-tet: nodes are all four
      unified-index particle indices.

    Newton's :attr:`Model.shape_body` uses Newton's body indexing
    (``[0, model.body_count)`` for dynamic bodies, ``-1`` for shapes
    anchored to the world). PhoenX's :class:`BodyContainer` may use
    a different layout: the "ported example" / cloth-aware convention
    reserves slot 0 for a static world-anchor body and shifts every
    Newton body by ``+1`` (so ``phoenx_body_offset = 1``); raw
    ``WorldBuilder`` scenes pass ``phoenx_body_offset = 0``.
    """
    s = wp.tid()
    if s < cloth_shape_offset:
        b = shape_body[s]
        if b >= 0:
            b = b + phoenx_body_offset
        ep = ShapeEndpoint()
        ep.nodes = wp.vec4i(b, wp.int32(-1), wp.int32(-1), wp.int32(-1))
        ep.kind = wp.int32(SHAPE_ENDPOINT_KIND_RIGID)
        shape_endpoints[s] = ep
        return
    if s < soft_tet_shape_offset:
        t = s - cloth_shape_offset
        if t >= num_cloth_triangles:
            return
        pa = tri_indices[t, 0]
        pb = tri_indices[t, 1]
        pc = tri_indices[t, 2]
        ep = ShapeEndpoint()
        ep.nodes = wp.vec4i(num_bodies + pa, num_bodies + pb, num_bodies + pc, wp.int32(-1))
        ep.kind = wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE)
        shape_endpoints[s] = ep
        return
    t = s - soft_tet_shape_offset
    if t >= num_soft_tetrahedra:
        return
    pa = tet_indices[t, 0]
    pb = tet_indices[t, 1]
    pc = tet_indices[t, 2]
    pd = tet_indices[t, 3]
    ep = ShapeEndpoint()
    ep.nodes = wp.vec4i(num_bodies + pa, num_bodies + pb, num_bodies + pc, num_bodies + pd)
    ep.kind = wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON)
    shape_endpoints[s] = ep


_DEGENERATE_EPS = wp.constant(wp.float32(1.0e-12))
_TET_SURFACE_BARY_EPS = wp.constant(wp.float32(1.0e-4))

# * `pack_contact_endpoints_kernel` -- per contact
#   *column*: re-stamps body1 / body2 to unified-index node[0] of each
#   side, fills side*_kind and side*_nodes_extra.
#
# * `pack_contact_barycentric_kernel` -- per individual
#   *contact* k: when a side is a cloth triangle, projects the
#   narrow-phase contact point onto the triangle plane and computes the
#   barycentric weights, stored in :class:`ContactContainer.lambdas`.
#
# Both kernels are no-ops when both sides are rigid (the kind tag short-
# circuits the cloth branch); rigid-only scenes simply don't launch
# them.


@wp.kernel(enable_backward=False)
def pack_contact_endpoints_kernel(
    pair_source_idx: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    shape_endpoints: wp.array[ShapeEndpoint],
    # out
    contact_cols: ContactColumnContainer,
):
    """Per contact column: stamp the cloth-aware endpoint metadata.

    Re-stamps :attr:`ContactConstraintData.body1` / :attr:`body2` with
    the unified body-or-particle index of each side's primary node
    (rigid -> rigid body unified-index; cloth -> first triangle
    particle's unified index). Then fills ``side*_kind`` and
    ``side*_nodes_extra`` so the iterate's endpoint helper has all the
    node info per side without re-touching the shape table.
    """
    tid = wp.tid()
    if tid >= num_contact_columns[0]:
        return

    p = pair_source_idx[tid]
    sa = pair_shape_a[p]
    sb = pair_shape_b[p]

    ep_a = shape_endpoints[sa]
    ep_b = shape_endpoints[sb]

    # Primary node per side (= nodes[0]) lands in the existing
    # ``body1`` / ``body2`` header dwords; extras (up to 3 for soft-tet,
    # ``-1`` for unused slots on rigid / cloth-tri) land in side*_nodes_extra.
    node_a = ep_a.nodes[0]
    node_b = ep_b.nodes[0]
    # Rigid-rigid columns use the fast rigid path, which expects valid body slots.
    # Mixed deformable rows keep -1 so endpoint helpers handle static anchors.
    if ep_a.kind == wp.int32(SHAPE_ENDPOINT_KIND_RIGID) and ep_b.kind == wp.int32(SHAPE_ENDPOINT_KIND_RIGID):
        if node_a < wp.int32(0):
            node_a = wp.int32(0)
        if node_b < wp.int32(0):
            node_b = wp.int32(0)
    contact_set_body1(contact_cols, tid, node_a)
    contact_set_body2(contact_cols, tid, node_b)
    contact_set_side0_kind(contact_cols, tid, ep_a.kind)
    contact_set_side1_kind(contact_cols, tid, ep_b.kind)
    contact_set_side0_nodes_extra(contact_cols, tid, wp.vec3i(ep_a.nodes[1], ep_a.nodes[2], ep_a.nodes[3]))
    contact_set_side1_nodes_extra(contact_cols, tid, wp.vec3i(ep_b.nodes[1], ep_b.nodes[2], ep_b.nodes[3]))


@wp.func
def _barycentric_in_tet(p: wp.vec3f, xa: wp.vec3f, xb: wp.vec3f, xc: wp.vec3f, xd: wp.vec3f) -> wp.vec3f:
    """Tet barycentric of point ``p`` w.r.t. vertices ``(xa, xb, xc, xd)``.

    Returns ``(bary_a, bary_b, bary_c)`` -- the weights for the first
    three vertices. The 4th weight is implicit:
    ``bary_d = 1 - bary_a - bary_b - bary_c`` (derived at iterate time
    by the 4-node contact endpoint helper).

    Mirrors :func:`_barycentric_in_plane` -- solve a 3x3 linear system
    on the edge basis ``(xb-xa, xc-xa, xd-xa)``. Degenerate (coplanar
    or collinear) tetrahedra collapse to ``(1, 0, 0)`` so the iterate
    sees a well-defined contact point at vertex A.
    """
    e1 = xb - xa
    e2 = xc - xa
    e3 = xd - xa
    d = p - xa
    # 3x3 matrix ``T`` with columns (e1, e2, e3); barycentric
    # (beta, gamma, delta) = T^-1 * d. ``alpha = 1 - beta - gamma - delta``.
    T = wp.mat33f(
        e1[0],
        e2[0],
        e3[0],
        e1[1],
        e2[1],
        e3[1],
        e1[2],
        e2[2],
        e3[2],
    )
    det_T = wp.determinant(T)
    if det_T < _DEGENERATE_EPS and det_T > -_DEGENERATE_EPS:
        return wp.vec3f(1.0, 0.0, 0.0)
    inv_T = wp.inverse(T)
    bcd = inv_T @ d
    beta = bcd[0]
    gamma = bcd[1]
    delta = bcd[2]
    alpha = wp.float32(1.0) - beta - gamma - delta

    # Contacts on a tetrahedron lie on its surface, so one barycentric
    # coordinate should be zero. Snap only the near-zero numerical case
    # and leave genuinely off-face points unchanged. The exact zero lets
    # the contact endpoint helpers skip the unused node.
    abs_alpha = wp.abs(alpha)
    abs_beta = wp.abs(beta)
    abs_gamma = wp.abs(gamma)
    abs_delta = wp.abs(delta)
    min_abs = abs_alpha
    drop = wp.int32(0)
    if abs_beta < min_abs:
        min_abs = abs_beta
        drop = wp.int32(1)
    if abs_gamma < min_abs:
        min_abs = abs_gamma
        drop = wp.int32(2)
    if abs_delta < min_abs:
        min_abs = abs_delta
        drop = wp.int32(3)

    if min_abs <= _TET_SURFACE_BARY_EPS:
        if drop == wp.int32(0):
            alpha = wp.float32(0.0)
        elif drop == wp.int32(1):
            beta = wp.float32(0.0)
        elif drop == wp.int32(2):
            gamma = wp.float32(0.0)
        else:
            delta = wp.float32(0.0)

        sum_kept = alpha + beta + gamma + delta
        if wp.abs(sum_kept) > _DEGENERATE_EPS:
            inv_sum = wp.float32(1.0) / sum_kept
            alpha = alpha * inv_sum
            beta = beta * inv_sum
            gamma = gamma * inv_sum

    return wp.vec3f(alpha, beta, gamma)


@wp.func
def _barycentric_in_plane(p: wp.vec3f, xa: wp.vec3f, xb: wp.vec3f, xc: wp.vec3f) -> wp.vec3f:
    """Project ``p`` onto the plane of triangle ``(xa, xb, xc)`` and
    return its barycentric weights ``(alpha, beta, gamma)`` with
    ``alpha + beta + gamma == 1`` (within float precision).

    Uses the standard 3x3 dot-product Cramer system on the in-plane
    edge basis; out-of-plane displacement is automatically removed by
    the projection (the in-plane gram matrix has rank 2).

    Returns ``(1, 0, 0)`` for degenerate triangles so subsequent reads
    don't NaN.
    """
    e1 = xb - xa
    e2 = xc - xa
    d = p - xa
    d00 = wp.dot(e1, e1)
    d01 = wp.dot(e1, e2)
    d11 = wp.dot(e2, e2)
    d20 = wp.dot(d, e1)
    d21 = wp.dot(d, e2)
    denom = d00 * d11 - d01 * d01
    if denom < _DEGENERATE_EPS and denom > -_DEGENERATE_EPS:
        return wp.vec3f(1.0, 0.0, 0.0)
    inv_denom = wp.float32(1.0) / denom
    beta = (d11 * d20 - d01 * d21) * inv_denom
    gamma = (d00 * d21 - d01 * d20) * inv_denom
    alpha = wp.float32(1.0) - beta - gamma
    return wp.vec3f(alpha, beta, gamma)


@wp.kernel(enable_backward=False)
def pack_contact_barycentric_kernel(
    contacts: ContactViews,
    shape_endpoints: wp.array[ShapeEndpoint],
    particles: ParticleContainer,
    num_bodies: wp.int32,
    # out
    cc: ContactContainer,
):
    """Per individual contact ``k``: when a side is a cloth triangle,
    compute the in-plane barycentric coords of the narrow-phase contact
    point against that side's three particle positions and store them
    in :class:`ContactContainer.lambdas`.

    Rigid sides leave ``side*_bary`` at zero (no-op). Inactive contact
    slots (``k >= rigid_contact_count``) early-return.

    The contact point used is the narrow-phase ``rigid_contact_point0``
    / ``rigid_contact_point1`` -- already in world space for cloth
    sides because :func:`CollisionPipeline._build_unified_shape_arrays`
    sets ``shape_body == -1`` for cloth shapes (forcing identity in the
    body-frame transform on the narrow-phase output).
    """
    k = wp.tid()
    # Clamp the count against the buffer capacity -- on narrow-phase
    # overflow ``rigid_contact_count[0]`` keeps climbing past the
    # actual buffer size while only the first ``rigid_contact_max``
    # slots got written. Without the clamp the early-return below
    # never fires for k in [0, buffer_size) and we'd read garbage
    # (or, downstream, OOB) from the tail.
    n_active = contacts.rigid_contact_count[0]
    if n_active > contacts.rigid_contact_shape0.shape[0]:
        n_active = contacts.rigid_contact_shape0.shape[0]
    if k >= n_active:
        return

    sa = contacts.rigid_contact_shape0[k]
    sb = contacts.rigid_contact_shape1[k]
    ep_a = shape_endpoints[sa]
    ep_b = shape_endpoints[sb]

    if ep_a.kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        p_a = ep_a.nodes[0] - num_bodies
        p_b = ep_a.nodes[1] - num_bodies
        p_c = ep_a.nodes[2] - num_bodies
        bary = _barycentric_in_plane(
            contacts.rigid_contact_point0[k],
            particles.position[p_a],
            particles.position[p_b],
            particles.position[p_c],
        )
        cc_set_side0_bary(cc, k, bary)
    elif ep_a.kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        p_a = ep_a.nodes[0] - num_bodies
        p_b = ep_a.nodes[1] - num_bodies
        p_c = ep_a.nodes[2] - num_bodies
        p_d = ep_a.nodes[3] - num_bodies
        bary = _barycentric_in_tet(
            contacts.rigid_contact_point0[k],
            particles.position[p_a],
            particles.position[p_b],
            particles.position[p_c],
            particles.position[p_d],
        )
        cc_set_side0_bary(cc, k, bary)

    if ep_b.kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        p_a = ep_b.nodes[0] - num_bodies
        p_b = ep_b.nodes[1] - num_bodies
        p_c = ep_b.nodes[2] - num_bodies
        bary = _barycentric_in_plane(
            contacts.rigid_contact_point1[k],
            particles.position[p_a],
            particles.position[p_b],
            particles.position[p_c],
        )
        cc_set_side1_bary(cc, k, bary)
    elif ep_b.kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        p_a = ep_b.nodes[0] - num_bodies
        p_b = ep_b.nodes[1] - num_bodies
        p_c = ep_b.nodes[2] - num_bodies
        p_d = ep_b.nodes[3] - num_bodies
        bary = _barycentric_in_tet(
            contacts.rigid_contact_point1[k],
            particles.position[p_a],
            particles.position[p_b],
            particles.position[p_c],
            particles.position[p_d],
        )
        cc_set_side1_bary(cc, k, bary)


# ---------------------------------------------------------------------------
# Broad-phase filter: drop cloth-tri pairs that share a particle node.
# ---------------------------------------------------------------------------
#
# Without this filter, every adjacent triangle pair in a cloth grid
# (~5-6 per triangle for a regular grid) generates a contact column,
