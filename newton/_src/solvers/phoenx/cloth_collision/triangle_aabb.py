# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-step AABB build for cloth triangles.

A phoenx scene with cloth collisions extends the
``narrow_phase.shape_aabb_lower / upper`` arrays from length ``S``
(rigid shape count) to length ``S + T`` (rigid + cloth triangles).
Slots ``[0, S)`` are filled by the standard rigid-shape AABB pass
in :func:`newton._src.sim.collide.compute_shape_aabbs`; this module
fills slots ``[S, S+T)`` from particle positions.

Each cloth triangle's AABB is the bounding box of its three vertex
positions, expanded uniformly by the particle radius (cloth
thickness) plus an explicit margin for contact-detection gap. The
expansion is symmetric so the same array doubles as both the
broadphase input (``shape_aabb_*`` consumed by SAP/NXN) and the
narrow-phase tight-AABB reference (``narrow_phase.shape_aabb_*``,
which reads the same external array when constructed with
``external_aabb=True``).
"""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def compute_cloth_triangle_aabbs_kernel(
    particle_q: wp.array[wp.vec3f],
    particle_radius: wp.array[wp.float32],
    tri_indices: wp.array[wp.vec3i],
    base_offset: wp.int32,
    extra_margin: wp.float32,
    # in / out: full-length [S + T] AABB arrays. The kernel only
    # writes slots in [base_offset, base_offset + tri_indices.shape[0]);
    # rigid AABBs in [0, base_offset) are populated by a separate pass.
    aabb_lower: wp.array[wp.vec3f],
    aabb_upper: wp.array[wp.vec3f],
):
    """Build the AABB of one cloth triangle.

    Launch dim = ``T``. Each thread handles one triangle ``t``,
    writing ``aabb_lower[base_offset + t]`` and ``aabb_upper[base_offset + t]``.
    The expansion factor is the *maximum* of the three vertex radii
    plus ``extra_margin``: the union of three sphere-swept points
    bounds the swept triangle (vertex radii are the cloth's
    half-thickness in Newton's convention).
    """
    t = wp.tid()
    idx = tri_indices[t]
    p0 = particle_q[idx[0]]
    p1 = particle_q[idx[1]]
    p2 = particle_q[idx[2]]

    lo = wp.vec3f(
        wp.min(wp.min(p0[0], p1[0]), p2[0]),
        wp.min(wp.min(p0[1], p1[1]), p2[1]),
        wp.min(wp.min(p0[2], p1[2]), p2[2]),
    )
    hi = wp.vec3f(
        wp.max(wp.max(p0[0], p1[0]), p2[0]),
        wp.max(wp.max(p0[1], p1[1]), p2[1]),
        wp.max(wp.max(p0[2], p1[2]), p2[2]),
    )

    r0 = particle_radius[idx[0]]
    r1 = particle_radius[idx[1]]
    r2 = particle_radius[idx[2]]
    pad = wp.max(wp.max(r0, r1), r2) + extra_margin
    pad_v = wp.vec3f(pad, pad, pad)

    aabb_lower[base_offset + t] = lo - pad_v
    aabb_upper[base_offset + t] = hi + pad_v


def launch_cloth_triangle_aabbs(
    particle_q: wp.array,
    particle_radius: wp.array,
    tri_indices: wp.array,
    base_offset: int,
    extra_margin: float,
    aabb_lower: wp.array,
    aabb_upper: wp.array,
    *,
    device: wp.DeviceLike = None,
) -> None:
    """Host launcher for :func:`compute_cloth_triangle_aabbs_kernel`.

    Args:
        particle_q: World-space particle positions, shape ``(num_particles, 3)``.
        particle_radius: Per-particle radius (cloth half-thickness),
            shape ``(num_particles,)``.
        tri_indices: Per-triangle vertex indices, shape ``(num_triangles, 3)``.
        base_offset: Where in ``aabb_lower / upper`` cloth triangle
            slots start. Equal to the number of rigid shapes ``S``.
        extra_margin: Additional contact-detection gap added to each
            triangle AABB on every face (e.g. PhoenX scene's contact gap).
        aabb_lower, aabb_upper: Output arrays of length ``S + T``.
            Only the cloth triangle slots are written.
        device: Warp device for the launch.
    """
    t = int(tri_indices.shape[0])
    if t == 0:
        return
    wp.launch(
        compute_cloth_triangle_aabbs_kernel,
        dim=t,
        inputs=[
            particle_q,
            particle_radius,
            tri_indices,
            wp.int32(base_offset),
            wp.float32(extra_margin),
            aabb_lower,
            aabb_upper,
        ],
        device=device,
    )
