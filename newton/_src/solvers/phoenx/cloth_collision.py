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

from newton._src.geometry.support_function import encode_vec3
from newton._src.solvers.phoenx.body import MOTION_STATIC
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
    "PhoenXClothShareVertexFilterData",
    "ShapeEndpoint",
    "_phoenx_pack_cloth_contact_barycentric_kernel",
    "_phoenx_pack_cloth_contact_endpoints_kernel",
    "_phoenx_populate_shape_endpoints_kernel",
    "_phoenx_update_cloth_shape_geometry_kernel",
    "_phoenx_update_soft_tet_shape_geometry_kernel",
    "build_phoenx_share_vertex_filter_data",
    "canonicalize_tetrahedron",
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
def _phoenx_populate_shape_endpoints_kernel(
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
        local_x[0],
        local_y[0],
        local_z[0],
        local_x[1],
        local_y[1],
        local_z[1],
        local_x[2],
        local_y[2],
        local_z[2],
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


@wp.func
def canonicalize_tetrahedron(xa: wp.vec3f, xb: wp.vec3f, xc: wp.vec3f, xd: wp.vec3f):
    """Map four world-space tet vertices to the canonical
    ``(shape_transform, edge_ab, c_y, c_z, d_local)`` form Newton's
    :data:`~newton.GeoType.TETRAHEDRON` consumes.

    The first three vertices (A, B, C) follow the same convention as
    :func:`canonicalize_triangle`. The 4th vertex ``D`` is transformed
    into the local frame so it can be packed into ``shape_source``.

    Returns ``(transform, edge_ab, c_y, c_z, d_local)``. Degenerate
    tetrahedra (collinear A/B or coplanar A/B/C) return identity
    transform + zero scale; the narrow phase sees a degenerate
    primitive and skips it cleanly.
    """
    xform, edge_ab, c_y, c_z = canonicalize_triangle(xa, xb, xc)
    if edge_ab < _DEGENERATE_EPS:
        return xform, edge_ab, c_y, c_z, wp.vec3f(0.0, 0.0, 0.0)
    # Local frame: A is the origin, +Z = (B-A)/|B-A|, +Y in-plane perp,
    # +X = +Y x +Z (face normal). D in local space is the inverse
    # transform applied to D's world position.
    d_local = wp.transform_point(wp.transform_inverse(xform), xd)
    return xform, edge_ab, c_y, c_z, d_local


@wp.kernel(enable_backward=False)
def _phoenx_update_soft_tet_shape_geometry_kernel(
    particles: ParticleContainer,
    tet_indices: wp.array2d[wp.int32],
    soft_tet_shape_offset: wp.int32,
    soft_body_thickness: wp.float32,
    soft_body_gap: wp.float32,
    # outputs
    geom_xform: wp.array[wp.transform],
    geom_data: wp.array[wp.vec4f],
    shape_source: wp.array[wp.uint64],
    aabb_lower: wp.array[wp.vec3f],
    aabb_upper: wp.array[wp.vec3f],
):
    """Per-step update of the soft-tet shape suffix.

    Mirrors :func:`_phoenx_update_cloth_shape_geometry_kernel` for the
    4-vertex tet primitive. One thread per tet. Writes the five
    per-step quantities Newton's collision pipeline reads for shape
    ``soft_tet_shape_offset + t``:

    * ``geom_xform`` -- canonical-frame world transform (A at origin,
      AB along local +Z, ABC in local YZ plane).
    * ``geom_data`` -- ``(edge_ab, c_y, c_z, thickness)``.
    * ``shape_source`` -- encoded vertex D in local frame
      (``encode_vec3``).
    * ``shape_aabb_lower/upper`` -- world AABB over the 4 vertices,
      enlarged by ``thickness + gap`` for speculative-contact detection.
    """
    t = wp.tid()
    s = soft_tet_shape_offset + t

    pa = tet_indices[t, 0]
    pb = tet_indices[t, 1]
    pc = tet_indices[t, 2]
    pd = tet_indices[t, 3]

    xa = particles.position[pa]
    xb = particles.position[pb]
    xc = particles.position[pc]
    xd = particles.position[pd]

    xform, edge_ab, c_y, c_z, d_local = canonicalize_tetrahedron(xa, xb, xc, xd)
    geom_xform[s] = xform
    geom_data[s] = wp.vec4f(edge_ab, c_y, c_z, soft_body_thickness)
    shape_source[s] = encode_vec3(d_local)

    enlargement = soft_body_thickness + soft_body_gap
    enlargement_vec = wp.vec3f(enlargement, enlargement, enlargement)
    lo = wp.vec3f(
        wp.min(wp.min(xa[0], xb[0]), wp.min(xc[0], xd[0])),
        wp.min(wp.min(xa[1], xb[1]), wp.min(xc[1], xd[1])),
        wp.min(wp.min(xa[2], xb[2]), wp.min(xc[2], xd[2])),
    )
    hi = wp.vec3f(
        wp.max(wp.max(xa[0], xb[0]), wp.max(xc[0], xd[0])),
        wp.max(wp.max(xa[1], xb[1]), wp.max(xc[1], xd[1])),
        wp.max(wp.max(xa[2], xb[2]), wp.max(xc[2], xd[2])),
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
    # ``body1`` / ``body2`` header dwords; extras (up to 3 for soft-tet,
    # ``-1`` for unused slots on rigid / cloth-tri) land in side*_nodes_extra.
    contact_set_body1(contact_cols, tid, ep_a.nodes[0])
    contact_set_body2(contact_cols, tid, ep_b.nodes[0])
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
            Shape indices ``< S`` are rigid (filter passes through);
            indices ``[S, S + num_cloth_triangles)`` are cloth tris
            with index ``t = shape - S`` into ``tri_indices``;
            indices ``[S + num_cloth_triangles, ...)`` are soft-tets
            with index ``t = shape - S - num_cloth_triangles`` into
            ``tet_indices``.
        num_cloth_triangles: Length of the cloth-tri block (separates
            cloth from soft-tet in the unified shape array).
        tri_indices: Per-triangle particle indices, shape
            ``(num_cloth_triangles, 3)``.
        tet_indices: Per-tet particle indices, shape
            ``(num_soft_tetrahedra, 4)``.
        sleeping_enabled: ``1`` when the sleeping pipeline is active --
            the filter drops rigid-rigid pairs where both shapes' bodies
            are "frozen" (sleeping, static, kinematic, or the world
            anchor). ``0`` skips the sleep check.
        phoenx_body_offset: Shift to apply to ``shape_body[s]`` before
            indexing ``body_island_root`` / ``body_motion_type`` (PhoenX
            slot 0 is the world anchor, so the offset is 1 in practice).
        shape_body: Per-shape Newton body index (``-1`` for world);
            length matches the unified shape array. Required when
            ``sleeping_enabled == 1``.
        body_island_root: PhoenX ``BodyContainer.island_root``. A body
            is sleeping iff its ``island_root >= 0``. Required when
            ``sleeping_enabled == 1``.
        body_motion_type: PhoenX ``BodyContainer.motion_type``. Required
            when ``sleeping_enabled == 1`` so the filter can treat
            STATIC / KINEMATIC bodies as permanently frozen.
    """

    num_rigid_shapes: wp.int32
    num_cloth_triangles: wp.int32
    tri_indices: wp.array2d[wp.int32]
    tet_indices: wp.array2d[wp.int32]
    sleeping_enabled: wp.int32
    phoenx_body_offset: wp.int32
    shape_body: wp.array[wp.int32]
    body_island_root: wp.array[wp.int32]
    body_motion_type: wp.array[wp.int32]


@wp.func
def _share_vertex_get_particles(
    shape: wp.int32,
    data: PhoenXClothShareVertexFilterData,
):
    """Return the up-to-4 particle indices of a deformable shape, or
    ``(-1, -1, -1, -1)`` if the shape is rigid. Used by
    :func:`phoenx_cloth_share_vertex_filter` to keep the filter body
    compact.
    """
    S = data.num_rigid_shapes
    if shape < S:
        return wp.int32(-1), wp.int32(-1), wp.int32(-1), wp.int32(-1)
    t_cloth = shape - S
    if t_cloth < data.num_cloth_triangles:
        return (
            data.tri_indices[t_cloth, 0],
            data.tri_indices[t_cloth, 1],
            data.tri_indices[t_cloth, 2],
            wp.int32(-1),
        )
    t_tet = t_cloth - data.num_cloth_triangles
    return (
        data.tet_indices[t_tet, 0],
        data.tet_indices[t_tet, 1],
        data.tet_indices[t_tet, 2],
        data.tet_indices[t_tet, 3],
    )


@wp.func
def phoenx_cloth_share_vertex_filter(
    pair: wp.vec2i,
    data: PhoenXClothShareVertexFilterData,
) -> wp.int32:
    """Broad-phase callback. Returns ``0`` to drop the pair, ``1`` to keep.

    Two independent filters compose here:

    1. (Sleeping) When the sleeping pipeline is active, drop any
       rigid-rigid pair where *both* shapes' bodies are "frozen": each
       is sleeping, attached to the world anchor (``shape_body == -1``),
       or has ``motion_type == STATIC``. KINEMATIC bodies do NOT count
       as frozen so a kinematic mover (e.g. a camera collider) still
       generates contacts against sleeping bricks and the sleeping
       pass wakes the impacted island next step.
    2. (Share-vertex) Drop a pair iff both shapes are deformable
       (cloth-tri or soft-tet) AND they share at least one particle
       vertex.
    """
    sa = pair[0]
    sb = pair[1]
    if data.sleeping_enabled != wp.int32(0):
        # Rigid prefix only -- deformable suffixes use shape_body=-1
        # via the cloth path, not the sleeping path.
        if sa < data.num_rigid_shapes and sb < data.num_rigid_shapes:
            nba = data.shape_body[sa]
            nbb = data.shape_body[sb]
            # Map shape_body == -1 (world anchor) to PhoenX slot 0; that
            # slot's motion_type is STATIC so the frozen test catches it.
            slot_a = wp.int32(0)
            slot_b = wp.int32(0)
            if nba >= 0:
                slot_a = nba + data.phoenx_body_offset
            if nbb >= 0:
                slot_b = nbb + data.phoenx_body_offset
            frozen_a = (data.body_island_root[slot_a] >= wp.int32(0)) or (
                data.body_motion_type[slot_a] == MOTION_STATIC
            )
            frozen_b = (data.body_island_root[slot_b] >= wp.int32(0)) or (
                data.body_motion_type[slot_b] == MOTION_STATIC
            )
            if frozen_a and frozen_b:
                return wp.int32(0)
    a0, a1, a2, a3 = _share_vertex_get_particles(sa, data)
    if a0 < wp.int32(0):
        return wp.int32(1)  # sa rigid: pass through
    b0, b1, b2, b3 = _share_vertex_get_particles(sb, data)
    if b0 < wp.int32(0):
        return wp.int32(1)  # sb rigid: pass through
    # Both deformable. Compare every pair of (active) vertex indices.
    for i in range(4):
        ai = a0
        if i == 1:
            ai = a1
        elif i == 2:
            ai = a2
        elif i == 3:
            ai = a3
        if ai < wp.int32(0):
            continue
        if ai == b0 or ai == b1 or ai == b2 or ai == b3:
            return wp.int32(0)
    return wp.int32(1)


def build_phoenx_share_vertex_filter_data(
    *,
    num_rigid_shapes: int,
    num_cloth_triangles: int,
    tri_indices: wp.array2d[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    sleeping_enabled: bool,
    phoenx_body_offset: int,
    shape_body: wp.array[wp.int32] | None,
    body_island_root: wp.array[wp.int32] | None,
    body_motion_type: wp.array[wp.int32] | None,
    device: wp.context.Devicelike,
) -> PhoenXClothShareVertexFilterData:
    """Construct the broad-phase filter data struct used by
    :func:`phoenx_cloth_share_vertex_filter`.

    Length-1 sentinel arrays are substituted for ``shape_body`` /
    ``body_island_root`` / ``body_motion_type`` when ``sleeping_enabled``
    is ``False`` so the Warp ABI stays bound regardless of feature use.
    The sentinel ``body_island_root`` is initialized to ``-1`` (awake)
    so the frozen test never trips when the field is unused.
    """
    data = PhoenXClothShareVertexFilterData()
    data.num_rigid_shapes = wp.int32(int(num_rigid_shapes))
    data.num_cloth_triangles = wp.int32(int(num_cloth_triangles))
    data.tri_indices = tri_indices
    data.tet_indices = tet_indices
    data.sleeping_enabled = wp.int32(1 if sleeping_enabled else 0)
    data.phoenx_body_offset = wp.int32(int(phoenx_body_offset))
    if shape_body is None:
        shape_body = wp.zeros(1, dtype=wp.int32, device=device)
    if body_island_root is None:
        body_island_root = wp.full(1, value=-1, dtype=wp.int32, device=device)
    if body_motion_type is None:
        body_motion_type = wp.zeros(1, dtype=wp.int32, device=device)
    data.shape_body = shape_body
    data.body_island_root = body_island_root
    data.body_motion_type = body_motion_type
    return data
