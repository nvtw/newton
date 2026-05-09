# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cloth-only substep kernels for :class:`PhoenXWorld`.

Self-contained pipeline that runs alongside (but independent of) the
rigid PGS pipeline. When :class:`PhoenXWorld` is constructed with
``num_particles > 0`` and ``num_cloth_triangles > 0`` and
:meth:`~PhoenXWorld.step` is called with ``contacts=None``, the cloth
path runs:

#. ``cloth_predict_kernel`` -- per particle: snapshot pose into
   ``position_prev_substep``, apply gravity to velocity, advance
   position.
#. Per iteration: per color group, ``cloth_iterate_kernel`` -- one
   thread per triangle in the group, runs
   :func:`cloth_triangle_iterate_at`. The host-side coloring guarantees
   no two triangles in the same group share a vertex, so the parallel
   writes to ``particles.position`` are race-free.
#. ``cloth_recover_kernel`` -- per particle: ``velocity = (position -
   position_prev_substep) * inv_dt``.

A single ``cloth_prepare_kernel`` runs once per substep before the
iteration loop to seed the per-triangle inverse-mass cache and zero
the XPBD lambda accumulators.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_STATIC,
    ACCESS_MODE_VELOCITY_LEVEL,
    synchronize_position_velocity,
)
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_triangle_iterate_at,
    cloth_triangle_prepare_for_iteration_at,
    cloth_triangle_set_alpha_lambda,
    cloth_triangle_set_alpha_mu,
    cloth_triangle_set_beta_lambda,
    cloth_triangle_set_beta_mu,
    cloth_triangle_set_body1,
    cloth_triangle_set_body2,
    cloth_triangle_set_body3,
    cloth_triangle_set_inv_rest,
    cloth_triangle_set_rest_area,
    cloth_triangle_set_type,
)
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "cloth_init_triangle_rows_kernel",
    "cloth_iterate_kernel",
    "cloth_predict_kernel",
    "cloth_prepare_kernel",
    "cloth_recover_kernel",
    "color_cloth_triangles",
]


_PHOENX_CLOTH_STIFFNESS_FLOOR = wp.constant(wp.float32(1.0e-6))


@wp.kernel
def cloth_predict_kernel(
    particles: ParticleContainer,
    gravity: wp.array[wp.vec3f],
    substep_dt: wp.float32,
):
    """Substep entry: snapshot pose, set access_mode, apply gravity, predict.

    The position is advanced to ``p + dt * v`` in the position field
    while ``access_mode = VELOCITY_LEVEL`` -- the
    :mod:`access_mode.synchronize_position_velocity` V->P branch
    integrates the same expression, so a subsequent constraint flip
    to ``POSITION_LEVEL`` is consistent with the predicted state.

    Particles with ``inverse_mass == 0`` get ``access_mode = STATIC``
    so synchronize is a no-op on them; the snapshot anchors them in
    place.
    """
    i = wp.tid()
    p = particles.position[i]
    particles.position_prev_substep[i] = p
    if particles.inverse_mass[i] == wp.float32(0.0):
        particles.access_mode[i] = ACCESS_MODE_STATIC
        return
    particles.access_mode[i] = ACCESS_MODE_VELOCITY_LEVEL
    v = particles.velocity[i] + gravity[0] * substep_dt
    particles.velocity[i] = v
    particles.position[i] = p + substep_dt * v


@wp.kernel
def cloth_recover_kernel(
    particles: ParticleContainer,
    inv_dt: wp.float32,
):
    """Substep exit: flip every particle to ``VELOCITY_LEVEL``.

    Routed through :func:`synchronize_position_velocity` so the
    finite-diff math stays in one place and ``STATIC`` particles are
    short-circuited consistently with the rest of the access-mode
    pattern.
    """
    i = wp.tid()
    pos_new, vel_new, mode_new = synchronize_position_velocity(
        particles.position[i],
        particles.velocity[i],
        particles.position_prev_substep[i],
        particles.access_mode[i],
        ACCESS_MODE_VELOCITY_LEVEL,
        inv_dt,
    )
    particles.position[i] = pos_new
    particles.velocity[i] = vel_new
    particles.access_mode[i] = mode_new


@wp.kernel
def cloth_prepare_kernel(
    constraints: ConstraintContainer,
    particles: ParticleContainer,
    cid_offset: wp.int32,
):
    """Per-triangle: cache inv_mass, zero lambda accumulators."""
    t = wp.tid()
    cloth_triangle_prepare_for_iteration_at(constraints, cid_offset + t, particles)


@wp.kernel
def cloth_iterate_kernel(
    constraints: ConstraintContainer,
    particles: ParticleContainer,
    color_cids: wp.array[wp.int32],
    idt: wp.float32,
):
    """Iterate every triangle in one color group in parallel."""
    t = wp.tid()
    cloth_triangle_iterate_at(constraints, color_cids[t], particles, idt)


@wp.kernel
def cloth_init_triangle_rows_kernel(
    constraints: ConstraintContainer,
    cid_offset: wp.int32,
    tri_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3f],
    tri_materials: wp.array2d[wp.float32],
    default_beta_lambda: wp.float32,
    default_beta_mu: wp.float32,
):
    """Stamp one cloth-triangle row from Newton mesh API.

    ``tri_materials[t, 0]`` is ``tri_ke`` (shear modulus mu, Pa);
    ``tri_materials[t, 1]`` is ``tri_ka`` (area Lame parameter lambda,
    Pa). XPBD compliance ``alpha = 1 / k`` (FemTriPBD.cs:60-61); area
    enters via the row gradients.
    """
    t = wp.tid()
    cid = cid_offset + t

    pa = tri_indices[t, 0]
    pb = tri_indices[t, 1]
    pc = tri_indices[t, 2]

    cloth_triangle_set_type(constraints, cid)
    cloth_triangle_set_body1(constraints, cid, pa)
    cloth_triangle_set_body2(constraints, cid, pb)
    cloth_triangle_set_body3(constraints, cid, pc)

    xa = particle_q[pa]
    xb = particle_q[pb]
    xc = particle_q[pc]
    ab = xb - xa
    ac = xc - xa
    normal = wp.cross(ab, ac)
    ab_len = wp.sqrt(wp.dot(ab, ab))
    if ab_len < wp.float32(1.0e-12):
        cloth_triangle_set_inv_rest(
            constraints, cid, wp.mat22f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
        )
        cloth_triangle_set_rest_area(constraints, cid, wp.float32(0.0))
    else:
        x_axis = ab / ab_len
        y_unnorm = wp.cross(normal, ab)
        y_len = wp.sqrt(wp.dot(y_unnorm, y_unnorm))
        if y_len < wp.float32(1.0e-12):
            cloth_triangle_set_inv_rest(
                constraints, cid, wp.mat22f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
            )
            cloth_triangle_set_rest_area(constraints, cid, wp.float32(0.0))
        else:
            y_axis = y_unnorm / y_len
            ab_x = wp.dot(x_axis, ab)
            ab_y = wp.dot(y_axis, ab)
            ac_x = wp.dot(x_axis, ac)
            ac_y = wp.dot(y_axis, ac)
            det = ab_x * ac_y - ab_y * ac_x
            if det < wp.float32(1.0e-12) and det > wp.float32(-1.0e-12):
                cloth_triangle_set_inv_rest(
                    constraints, cid, wp.mat22f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
                )
                cloth_triangle_set_rest_area(constraints, cid, wp.float32(0.0))
            else:
                inv_det = wp.float32(1.0) / det
                inv_rest_m = wp.mat22f(
                    ac_y * inv_det, -ac_x * inv_det,
                    -ab_y * inv_det, ab_x * inv_det,
                )
                cloth_triangle_set_inv_rest(constraints, cid, inv_rest_m)
                rest_area = wp.float32(0.5) * wp.sqrt(wp.dot(normal, normal))
                cloth_triangle_set_rest_area(constraints, cid, rest_area)

    k_mu = tri_materials[t, 0]
    if k_mu < _PHOENX_CLOTH_STIFFNESS_FLOOR:
        k_mu = _PHOENX_CLOTH_STIFFNESS_FLOOR
    k_lambda = tri_materials[t, 1]
    if k_lambda < _PHOENX_CLOTH_STIFFNESS_FLOOR:
        k_lambda = _PHOENX_CLOTH_STIFFNESS_FLOOR
    cloth_triangle_set_alpha_lambda(constraints, cid, wp.float32(1.0) / k_lambda)
    cloth_triangle_set_alpha_mu(constraints, cid, wp.float32(1.0) / k_mu)
    cloth_triangle_set_beta_lambda(constraints, cid, default_beta_lambda)
    cloth_triangle_set_beta_mu(constraints, cid, default_beta_mu)


def color_cloth_triangles(tri_indices, num_particles: int) -> list[list[int]]:
    """Greedy graph coloring on triangles: two triangles share a color
    only if they touch no common particle.

    Args:
        tri_indices: ``np.ndarray`` of shape ``(num_tris, 3)``, host-side.
        num_particles: Particle count (for the per-particle adjacency
            scratch).

    Returns:
        List of color groups; each group is a list of triangle ids
        (cids relative to the cloth block, not absolute).
    """
    import numpy as np  # noqa: PLC0415

    tri = np.asarray(tri_indices, dtype=np.int64)
    num_tris = int(tri.shape[0])
    if num_tris == 0:
        return []

    particle_tris: list[list[int]] = [[] for _ in range(num_particles)]
    for t in range(num_tris):
        for k in range(3):
            particle_tris[int(tri[t, k])].append(t)

    color = [-1] * num_tris
    for t in range(num_tris):
        used: set[int] = set()
        for k in range(3):
            v = int(tri[t, k])
            for nt in particle_tris[v]:
                if nt != t and color[nt] >= 0:
                    used.add(color[nt])
        c = 0
        while c in used:
            c += 1
        color[t] = c

    num_colors = max(color) + 1
    groups: list[list[int]] = [[] for _ in range(num_colors)]
    for t, c in enumerate(color):
        groups[c].append(t)
    return groups
