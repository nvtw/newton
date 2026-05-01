# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-step cloth-triangle stamp kernel (fused AABB + shape data).

Single per-cloth-triangle pass that fills both:

* the broadphase AABB slots ``aabb_lower / upper[S + t]``, and
* the narrowphase shape data slots ``shape_type / shape_transform /
  shape_data / shape_auxiliary[S + t]``.

The two writes share the same per-triangle reads of ``particle_q``
and ``tri_indices``, so fusing them into a single kernel halves the
per-step launch count compared with running the AABB pass and the
shape-data pass separately.
"""

from __future__ import annotations

import warp as wp

from newton._src.geometry.support_function import GeoTypeEx
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_triangle_get_body1,
    cloth_triangle_get_body2,
    cloth_triangle_get_body3,
)
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer


@wp.kernel(enable_backward=False)
def compute_cloth_triangle_stamp_kernel(
    particle_q: wp.array[wp.vec3f],
    particle_radius: wp.array[wp.float32],
    # Topology source: cloth-triangle rows in the phoenx
    # ConstraintContainer. Triangle ``t`` lives at ``cid =
    # cloth_cid_offset + t`` and stores its three vertex indices
    # in the body1/body2/body3 slots as *unified* body-or-particle
    # indices. ``num_bodies`` is the unified-index offset between
    # rigid bodies and particles (subtract to recover the raw
    # particle index for ``particle_q``).
    constraints: ConstraintContainer,
    cloth_cid_offset: wp.int32,
    num_bodies: wp.int32,
    base_offset: wp.int32,
    aabb_extra_margin: wp.float32,
    shape_data_margin: wp.float32,
    # in / out: full-length [S + T] arrays. Only slots in
    # [base_offset, base_offset + T) are written.
    shape_type: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_data: wp.array[wp.vec4],
    shape_auxiliary: wp.array[wp.vec3f],
    aabb_lower: wp.array[wp.vec3f],
    aabb_upper: wp.array[wp.vec3f],
):
    """Stamp one cloth triangle into the unified shape + AABB arrays.

    Launch dim = ``T``.  Each thread reads the three particle
    positions of triangle ``t`` once and writes:

    * ``shape_type[s] = GeoTypeEx.TRIANGLE``
    * ``shape_transform[s] = (vertex A in world space, identity quat)``
    * ``shape_data[s] = (B - A, shape_data_margin)``
    * ``shape_auxiliary[s] = C - A``
    * ``aabb_lower / upper[s] = bbox(A, B, C) +/- (max_radius + aabb_extra_margin)``
    """
    t = wp.tid()
    cid = cloth_cid_offset + t
    ia = cloth_triangle_get_body1(constraints, cid) - num_bodies
    ib = cloth_triangle_get_body2(constraints, cid) - num_bodies
    ic = cloth_triangle_get_body3(constraints, cid) - num_bodies
    pa = particle_q[ia]
    pb = particle_q[ib]
    pc = particle_q[ic]
    s = base_offset + t

    # Shape descriptor: vertex A as world-space origin, B/C as
    # local-frame offsets.
    shape_type[s] = wp.int32(int(GeoTypeEx.TRIANGLE))
    shape_transform[s] = wp.transform(pa, wp.quat_identity())
    ab = pb - pa
    ac = pc - pa
    shape_data[s] = wp.vec4(ab[0], ab[1], ab[2], shape_data_margin)
    shape_auxiliary[s] = ac

    # AABB: bbox of (A, B, C) expanded by max(per-vertex radius) +
    # extra contact-detection margin.
    lo = wp.vec3f(
        wp.min(wp.min(pa[0], pb[0]), pc[0]),
        wp.min(wp.min(pa[1], pb[1]), pc[1]),
        wp.min(wp.min(pa[2], pb[2]), pc[2]),
    )
    hi = wp.vec3f(
        wp.max(wp.max(pa[0], pb[0]), pc[0]),
        wp.max(wp.max(pa[1], pb[1]), pc[1]),
        wp.max(wp.max(pa[2], pb[2]), pc[2]),
    )
    r0 = particle_radius[ia]
    r1 = particle_radius[ib]
    r2 = particle_radius[ic]
    pad = wp.max(wp.max(r0, r1), r2) + aabb_extra_margin
    pad_v = wp.vec3f(pad, pad, pad)
    aabb_lower[s] = lo - pad_v
    aabb_upper[s] = hi + pad_v


def launch_cloth_triangle_stamp(
    particle_q: wp.array,
    particle_radius: wp.array,
    constraints: ConstraintContainer,
    cloth_cid_offset: int,
    num_bodies: int,
    num_cloth_triangles: int,
    base_offset: int,
    aabb_extra_margin: float,
    shape_data_margin: float,
    shape_type: wp.array,
    shape_transform: wp.array,
    shape_data: wp.array,
    shape_auxiliary: wp.array,
    aabb_lower: wp.array,
    aabb_upper: wp.array,
    *,
    device: wp.DeviceLike = None,
) -> None:
    """Host launcher for :func:`compute_cloth_triangle_stamp_kernel`.

    Reads triangle topology from the cloth-triangle rows of the
    phoenx :class:`ConstraintContainer` (single source of truth --
    no separate ``tri_indices`` buffer is maintained).
    """
    if num_cloth_triangles == 0:
        return
    wp.launch(
        compute_cloth_triangle_stamp_kernel,
        dim=num_cloth_triangles,
        inputs=[
            particle_q,
            particle_radius,
            constraints,
            wp.int32(cloth_cid_offset),
            wp.int32(num_bodies),
            wp.int32(base_offset),
            wp.float32(aabb_extra_margin),
            wp.float32(shape_data_margin),
            shape_type,
            shape_transform,
            shape_data,
            shape_auxiliary,
            aabb_lower,
            aabb_upper,
        ],
        device=device,
    )
