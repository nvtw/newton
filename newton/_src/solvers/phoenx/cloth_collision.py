# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cloth-triangle collision plumbing -- per-step shape geometry update.

PhoenX cloth-tris reuse Newton's existing rigid-triangle narrow-phase code
path verbatim. The only "novel" piece is that cloth-tri vertices move
every step (the cloth deforms), so the canonical local-frame triangle
encoding ``(shape_transform, shape_scale = (edge_ab, c_y, c_z))`` has to
be re-derived from the current particle positions before each collision
detection pass.

Layout in the unified shape arrays (managed by
:class:`~newton._src.sim.collide.CollisionPipeline` via
``extra_shape_count > 0``):

* Rigid shape prefix at ``[0, S)`` -- written by the existing
  :func:`compute_shape_aabbs` per step.
* Cloth-triangle shape suffix at ``[S, S + T)`` -- written by
  :func:`_phoenx_update_cloth_shape_geometry_kernel` here, once per step,
  immediately before :meth:`CollisionPipeline.collide_with_external_aabbs`.

The kernel below writes the four per-step suffix arrays
(``geom_transform``, ``geom_data``, ``shape_aabb_lower``,
``shape_aabb_upper``) using the same conventions
:func:`compute_shape_aabbs` uses for rigid triangles, so the broad and
narrow phases see one homogeneous shape population and dispatch on
``GeoType.TRIANGLE`` for cloth tris exactly as for rigid tris.
"""

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
    "PhoenXClothShareVertexFilterData",
    "SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE",
    "SHAPE_ENDPOINT_KIND_RIGID",
    "ShapeEndpoint",
    "_phoenx_pack_cloth_contact_barycentric_kernel",
    "_phoenx_pack_cloth_contact_endpoints_kernel",
    "_phoenx_populate_shape_endpoints_kernel",
    "_phoenx_update_cloth_shape_geometry_kernel",
    "canonicalize_triangle",
    "phoenx_cloth_share_vertex_filter",
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
# :meth:`PhoenXWorld.setup_cloth_collision_pipeline` and read by the
# contact-ingest kernel as a single 16-byte load per shape.

#: Shape is owned by a rigid body. ``nodes`` holds ``(body_unified, -1, -1)``.
SHAPE_ENDPOINT_KIND_RIGID: int = 0

#: Shape is a cloth triangle. ``nodes`` holds three particle indices in
#: unified body-or-particle space ``(num_bodies + p_a, num_bodies + p_b,
#: num_bodies + p_c)``.
SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE: int = 1


@wp.struct
class ShapeEndpoint:
    """Per-shape mapping ``shape_index -> (kind, nodes)``.

    For rigid shapes, ``nodes[0]`` is the rigid body index in unified
    body-or-particle space (``[0, num_bodies)``); ``nodes[1]`` and
    ``nodes[2]`` are ``-1``.

    For cloth-triangle shapes, ``nodes[0/1/2]`` are the three triangle
    particle indices in unified space (``num_bodies + particle_id``).
    """

    nodes: wp.vec3i
    kind: wp.int32


def shape_endpoints_zeros(num_shapes: int, device=None) -> wp.array[ShapeEndpoint]:
    """Allocate a zero-initialised :class:`ShapeEndpoint` array of
    length ``num_shapes``."""
    return wp.zeros(int(num_shapes), dtype=ShapeEndpoint, device=device)


@wp.kernel(enable_backward=False)
def _phoenx_populate_shape_endpoints_kernel(
    shape_body: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    cloth_shape_offset: wp.int32,
    num_cloth_triangles: wp.int32,
    num_bodies: wp.int32,
    phoenx_body_offset: wp.int32,
    # out
    shape_endpoints: wp.array[ShapeEndpoint],
):
    """One thread per shape ``s`` in ``[0, S + T)``: stamp its
    :class:`ShapeEndpoint`.

    * ``s < cloth_shape_offset`` -> rigid: ``nodes = (newton_body +
      phoenx_body_offset, -1, -1)``.

      Newton's :attr:`Model.shape_body` uses Newton's body indexing
      (``[0, model.body_count)`` for dynamic bodies, ``-1`` for shapes
      anchored to the world). PhoenX's :class:`BodyContainer` may use
      a different layout: the "ported example" / cloth-aware convention
      reserves slot 0 for a static world-anchor body and shifts every
      Newton body by ``+1`` (so ``phoenx_body_offset = 1``); raw
      ``WorldBuilder`` scenes pass ``phoenx_body_offset = 0``.
      ``-1`` (no body) is preserved as-is so the iterate's static-anchor
      branch fires.
    * ``s >= cloth_shape_offset`` -> cloth tri: nodes = unified-index
      triplet ``num_bodies + particle_id`` from ``tri_indices[s - S]``.
    """
    s = wp.tid()
    if s < cloth_shape_offset:
        b = shape_body[s]
        if b >= 0:
            b = b + phoenx_body_offset
        ep = ShapeEndpoint()
        ep.nodes = wp.vec3i(b, wp.int32(-1), wp.int32(-1))
        ep.kind = wp.int32(SHAPE_ENDPOINT_KIND_RIGID)
        shape_endpoints[s] = ep
        return
    t = s - cloth_shape_offset
    if t >= num_cloth_triangles:
        return
    pa = tri_indices[t, 0]
    pb = tri_indices[t, 1]
    pc = tri_indices[t, 2]
    ep = ShapeEndpoint()
    ep.nodes = wp.vec3i(num_bodies + pa, num_bodies + pb, num_bodies + pc)
    ep.kind = wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE)
    shape_endpoints[s] = ep


_DEGENERATE_EPS = wp.constant(wp.float32(1.0e-12))


@wp.func
def canonicalize_triangle(xa: wp.vec3f, xb: wp.vec3f, xc: wp.vec3f):
    """Map three world-space triangle vertices to the canonical
    ``(shape_transform, edge_ab, c_y, c_z)`` Newton's
    :data:`~newton.GeoType.TRIANGLE` consumes.

    Direct port of :func:`newton._src.sim.builder._canonicalize_triangle`.

    Local-frame convention (matches the host):

    * Vertex A at the local origin.
    * Local +Z = ``(B - A) / |B - A|``; vertex B sits at
      ``(0, 0, edge_ab)`` in local coords.
    * Local +Y = in-plane perpendicular to +Z, oriented so ``c_y > 0``;
      vertex C sits at ``(0, c_y, c_z)`` in local coords.
    * Local +X = ``+Y x +Z`` (right-handed); the triangle's face normal
      lies along local +X.

    Returns ``(transform, edge_ab, c_y, c_z)``. Degenerate triangles
    (collinear or zero edge) return an identity transform with zero
    scale -- the narrow phase then sees a degenerate triangle and
    skips it cleanly.
    """
    ab = xb - xa
    ac = xc - xa
    edge_ab_sq = wp.dot(ab, ab)
    if edge_ab_sq < _DEGENERATE_EPS:
        return wp.transform_identity(), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
    edge_ab = wp.sqrt(edge_ab_sq)
    local_z = ab * (wp.float32(1.0) / edge_ab)
    c_z = wp.dot(ac, local_z)
    perp = ac - c_z * local_z
    perp_norm_sq = wp.dot(perp, perp)
    if perp_norm_sq < _DEGENERATE_EPS:
        return wp.transform_identity(), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
    c_y = wp.sqrt(perp_norm_sq)
    local_y = perp * (wp.float32(1.0) / c_y)
    local_x = wp.cross(local_y, local_z)
    rot = wp.mat33f(
        local_x[0], local_y[0], local_z[0],
        local_x[1], local_y[1], local_z[1],
        local_x[2], local_y[2], local_z[2],
    )
    q = wp.quat_from_matrix(rot)
    return wp.transform(xa, q), edge_ab, c_y, c_z


@wp.kernel(enable_backward=False)
def _phoenx_update_cloth_shape_geometry_kernel(
    particles: ParticleContainer,
    tri_indices: wp.array2d[wp.int32],
    cloth_shape_offset: wp.int32,
    cloth_thickness: wp.float32,
    cloth_gap: wp.float32,
    # outputs
    geom_xform: wp.array[wp.transform],
    geom_data: wp.array[wp.vec4f],
    aabb_lower: wp.array[wp.vec3f],
    aabb_upper: wp.array[wp.vec3f],
):
    """Per-step update of the cloth-triangle shape suffix.

    One thread per cloth triangle. Writes the four per-step quantities
    Newton's collision pipeline reads for shape ``S + t``:

    * ``geom_xform[S+t]`` -- world transform from the canonical
      local-frame triangle (origin at vertex A, +Z along AB).
    * ``geom_data[S+t]`` -- ``(edge_ab, c_y, c_z, thickness)``; the
      narrow phase treats ``thickness`` as the Minkowski skin radius
      around the triangle.
    * ``shape_aabb_lower/upper[S+t]`` -- world AABB enlarged by
      ``thickness + gap`` (the standard Newton speculative-contact
      enlargement convention).
    """
    t = wp.tid()
    s = cloth_shape_offset + t

    pa = tri_indices[t, 0]
    pb = tri_indices[t, 1]
    pc = tri_indices[t, 2]

    xa = particles.position[pa]
    xb = particles.position[pb]
    xc = particles.position[pc]

    xform, edge_ab, c_y, c_z = canonicalize_triangle(xa, xb, xc)
    geom_xform[s] = xform
    geom_data[s] = wp.vec4f(edge_ab, c_y, c_z, cloth_thickness)

    # World AABB from the 3 vertices, enlarged by (thickness + gap)
    # for speculative-contact detection. Tighter than the support-map
    # AABB compute_shape_aabbs uses for rigid triangles, but the broad
    # phase only requires a conservative bound -- this one is.
    enlargement = cloth_thickness + cloth_gap
    enlargement_vec = wp.vec3f(enlargement, enlargement, enlargement)
    lo = wp.vec3f(
        wp.min(xa[0], wp.min(xb[0], xc[0])),
        wp.min(xa[1], wp.min(xb[1], xc[1])),
        wp.min(xa[2], wp.min(xb[2], xc[2])),
    )
    hi = wp.vec3f(
        wp.max(xa[0], wp.max(xb[0], xc[0])),
        wp.max(xa[1], wp.max(xb[1], xc[1])),
        wp.max(xa[2], wp.max(xb[2], xc[2])),
    )
    aabb_lower[s] = lo - enlargement_vec
    aabb_upper[s] = hi + enlargement_vec


# ---------------------------------------------------------------------------
# Cloth-aware contact ingest (post-process to the rigid pack kernel)
# ---------------------------------------------------------------------------
#
# These two kernels run after the standard contact ingest pipeline has
# materialised the contact columns from the rigid-only path. They
# overlay the cloth-aware fields:
#
# * `_phoenx_pack_cloth_contact_endpoints_kernel` -- per contact
#   *column*: re-stamps body1 / body2 to unified-index node[0] of each
#   side, fills side*_kind and side*_nodes_extra.
#
# * `_phoenx_pack_cloth_contact_barycentric_kernel` -- per individual
#   *contact* k: when a side is a cloth triangle, projects the
#   narrow-phase contact point onto the triangle plane and computes the
#   barycentric weights, stored in :class:`ContactContainer.lambdas`.
#
# Both kernels are no-ops when both sides are rigid (the kind tag short-
# circuits the cloth branch); rigid-only scenes simply don't launch
# them.


@wp.kernel(enable_backward=False)
def _phoenx_pack_cloth_contact_endpoints_kernel(
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
    # ``body1`` / ``body2`` header dwords; extras land in side*_nodes_extra.
    contact_set_body1(contact_cols, tid, ep_a.nodes[0])
    contact_set_body2(contact_cols, tid, ep_b.nodes[0])
    contact_set_side0_kind(contact_cols, tid, ep_a.kind)
    contact_set_side1_kind(contact_cols, tid, ep_b.kind)
    contact_set_side0_nodes_extra(contact_cols, tid, wp.vec2i(ep_a.nodes[1], ep_a.nodes[2]))
    contact_set_side1_nodes_extra(contact_cols, tid, wp.vec2i(ep_b.nodes[1], ep_b.nodes[2]))


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
def _phoenx_pack_cloth_contact_barycentric_kernel(
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
    if k >= contacts.rigid_contact_count[0]:
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


# ---------------------------------------------------------------------------
# Broad-phase filter: drop cloth-tri pairs that share a particle node.
# ---------------------------------------------------------------------------
#
# Without this filter, every adjacent triangle pair in a cloth grid
# (~5-6 per triangle for a regular grid) generates a contact column,
# which is both wasteful and physically incorrect: triangles connected
# via a shared vertex are already linked by the cloth's XPBD elasticity
# constraints; treating their geometric overlap as a contact would
# double-count the same constraint.
#
# Rigid-rigid pairs and cloth-vs-rigid pairs pass through unchanged --
# the filter only fires on pairs where *both* shapes are cloth tris.
# Cloth-cloth pairs that don't share a vertex (real self-collision in
# a folded cloth) also pass through.
#
# Installed on the broad phase via the ``broad_phase_filter`` arg of
# :class:`CollisionPipeline` (see :meth:`PhoenXWorld.setup_cloth_collision_pipeline`).


@wp.struct
class PhoenXClothShareVertexFilterData:
    """Runtime data for :func:`phoenx_cloth_share_vertex_filter`.

    Attributes:
        num_rigid_shapes: ``S`` in the unified shape index space.
            Shape indices ``< num_rigid_shapes`` are rigid (filter
            passes through); indices ``>= num_rigid_shapes`` are cloth
            tris with index ``t = shape - S`` into ``tri_indices``.
        tri_indices: Per-triangle particle indices, shape
            ``(num_cloth_triangles, 3)``. Read at broad-phase time to
            test whether two cloth tris share a vertex.
    """

    num_rigid_shapes: wp.int32
    tri_indices: wp.array2d[wp.int32]


@wp.func
def phoenx_cloth_share_vertex_filter(
    pair: wp.vec2i,
    data: PhoenXClothShareVertexFilterData,
) -> wp.int32:
    """Broad-phase callback. Returns ``0`` to drop the pair, ``1`` to keep.

    Drop iff both shapes are cloth tris AND they share at least one
    vertex (any of the 3 particle indices match between the two
    triangles). Otherwise pass through.
    """
    sa = pair[0]
    sb = pair[1]
    S = data.num_rigid_shapes
    if sa < S or sb < S:
        return wp.int32(1)  # not both cloth: pass through
    ta = sa - S
    tb = sb - S
    a0 = data.tri_indices[ta, 0]
    a1 = data.tri_indices[ta, 1]
    a2 = data.tri_indices[ta, 2]
    b0 = data.tri_indices[tb, 0]
    b1 = data.tri_indices[tb, 1]
    b2 = data.tri_indices[tb, 2]
    if a0 == b0 or a0 == b1 or a0 == b2:
        return wp.int32(0)
    if a1 == b0 or a1 == b1 or a1 == b2:
        return wp.int32(0)
    if a2 == b0 or a2 == b1 or a2 == b2:
        return wp.int32(0)
    return wp.int32(1)
