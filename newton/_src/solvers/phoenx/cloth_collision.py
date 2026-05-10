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

from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "_phoenx_update_cloth_shape_geometry_kernel",
    "canonicalize_triangle",
]


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
