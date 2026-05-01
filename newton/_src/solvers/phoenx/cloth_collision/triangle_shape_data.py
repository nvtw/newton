# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-step stamping of cloth triangles into the unified shape arrays.

Cloth triangles flow through Newton's standard collision pipeline as
*virtual shapes* of type :data:`GeoTypeEx.TRIANGLE`. The shape index
space is concatenated: rigid shapes occupy ``[0, S)`` and cloth
triangles occupy ``[S, S+T)``. Each per-step pass writes the
following slots for ``s = S + t`` (``t`` in ``[0, T)``):

* ``shape_type[s] = GeoTypeEx.TRIANGLE``
* ``shape_transform[s] = transform(particle_q[ia], identity_quat)``
  -- the world-space position of vertex A is the triangle's "shape
  origin", with identity rotation. Vertices B and C live in the
  shape-local frame as offsets from A.
* ``shape_data[s] = vec4(B - A, margin)`` -- the local-frame offset
  to vertex B in xyz, margin in w (to satisfy the existing
  ``extract_shape_data`` contract that treats slot ``w`` as the
  margin offset).
* ``shape_auxiliary[s] = C - A`` -- the local-frame offset to vertex
  C. This is a new parallel array (length ``S+T``); slots ``[0, S)``
  are unused.

Cadence: collision detection runs once per *full step*, not per
substep, so the triangle pose can be sampled from particle positions
at step boundaries. (If sub-step collision detection is ever added,
this kernel would move to the per-substep schedule.)

Alternative considered (and rejected): bypass
:func:`extract_shape_data` for cloth triangles and build the
``GenericShapeData`` inline in the narrow phase from
``(particle_q, tri_indices, shape_idx - S)``. That avoids the new
``shape_auxiliary`` array but adds a triangle-aware branch into
every narrowphase path that touches a triangle shape, which is more
invasive than the parallel-array approach taken here.
"""

from __future__ import annotations

import warp as wp

from newton._src.geometry.support_function import GeoTypeEx


@wp.kernel(enable_backward=False)
def compute_cloth_triangle_shape_data_kernel(
    particle_q: wp.array[wp.vec3f],
    tri_indices: wp.array[wp.vec3i],
    base_offset: wp.int32,
    margin: wp.float32,
    # in / out: full-length [S + T] shape arrays. The kernel only
    # writes slots in [base_offset, base_offset + tri_indices.shape[0]).
    shape_type: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_data: wp.array[wp.vec4],
    shape_auxiliary: wp.array[wp.vec3f],
):
    """Stamp one cloth triangle's GenericShapeData equivalents.

    Launch dim = ``T``. Each thread handles one triangle ``t``,
    writing the four shape arrays at slot ``base_offset + t``. The
    triangle's local frame puts vertex A at the origin with identity
    rotation; B and C are stored as offsets from A in
    ``shape_data[3]`` (xyz) and ``shape_auxiliary``.

    The ``margin`` value is forwarded into ``shape_data[w]`` so the
    standard ``extract_shape_data`` contract (xyz = scale, w = margin)
    holds verbatim.
    """
    t = wp.tid()
    idx = tri_indices[t]
    pa = particle_q[idx[0]]
    pb = particle_q[idx[1]]
    pc = particle_q[idx[2]]

    s = base_offset + t

    # Position = vertex A in world space; orientation = identity.
    shape_type[s] = wp.int32(int(GeoTypeEx.TRIANGLE))
    shape_transform[s] = wp.transform(pa, wp.quat_identity())

    # Local-frame offsets: B - A in shape_data.xyz, C - A in shape_auxiliary.
    # ``shape_data.w`` keeps the per-shape margin offset that
    # ``extract_shape_data`` returns to the narrow phase.
    ab = pb - pa
    ac = pc - pa
    shape_data[s] = wp.vec4(ab[0], ab[1], ab[2], margin)
    shape_auxiliary[s] = ac


def launch_cloth_triangle_shape_data(
    particle_q: wp.array,
    tri_indices: wp.array,
    base_offset: int,
    margin: float,
    shape_type: wp.array,
    shape_transform: wp.array,
    shape_data: wp.array,
    shape_auxiliary: wp.array,
    *,
    device: wp.DeviceLike = None,
) -> None:
    """Host launcher for :func:`compute_cloth_triangle_shape_data_kernel`.

    Args:
        particle_q: World-space particle positions, shape
            ``(num_particles, 3)``.
        tri_indices: Per-triangle vertex indices, shape
            ``(num_triangles, 3)``.
        base_offset: Where in the unified shape arrays cloth triangle
            slots start. Equal to the number of rigid shapes ``S``.
        margin: Per-shape contact-detection gap forwarded into
            ``shape_data[w]`` for every cloth triangle slot.
        shape_type, shape_transform, shape_data, shape_auxiliary:
            Output arrays of length ``S + T``. Only the cloth
            triangle slots are written.
        device: Warp device for the launch.
    """
    t = int(tri_indices.shape[0])
    if t == 0:
        return
    wp.launch(
        compute_cloth_triangle_shape_data_kernel,
        dim=t,
        inputs=[
            particle_q,
            tri_indices,
            wp.int32(base_offset),
            wp.float32(margin),
            shape_type,
            shape_transform,
            shape_data,
            shape_auxiliary,
        ],
        device=device,
    )
