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


@wp.kernel(enable_backward=False)
def compute_cloth_triangle_stamp_kernel(
    particle_q: wp.array[wp.vec3f],
    particle_radius: wp.array[wp.float32],
    tri_indices: wp.array[wp.vec3i],
    base_offset: wp.int32,
    aabb_extra_margin: wp.float32,
    shape_data_margin: wp.float32,
    # in / out: full-length [S + T] arrays. Only slots in
    # [base_offset, base_offset + tri_indices.shape[0]) are written.
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
    idx = tri_indices[t]
    pa = particle_q[idx[0]]
    pb = particle_q[idx[1]]
    pc = particle_q[idx[2]]
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
    r0 = particle_radius[idx[0]]
    r1 = particle_radius[idx[1]]
    r2 = particle_radius[idx[2]]
    pad = wp.max(wp.max(r0, r1), r2) + aabb_extra_margin
    pad_v = wp.vec3f(pad, pad, pad)
    aabb_lower[s] = lo - pad_v
    aabb_upper[s] = hi + pad_v


def launch_cloth_triangle_stamp(
    particle_q: wp.array,
    particle_radius: wp.array,
    tri_indices: wp.array,
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

    Replaces the previously separate
    :func:`launch_cloth_triangle_aabbs` +
    :func:`launch_cloth_triangle_shape_data` calls with a single
    fused launch; the per-step cloth-collision overhead drops from
    two kernel launches to one.
    """
    t = int(tri_indices.shape[0])
    if t == 0:
        return
    wp.launch(
        compute_cloth_triangle_stamp_kernel,
        dim=t,
        inputs=[
            particle_q,
            particle_radius,
            tri_indices,
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
