# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cloth substep entry / exit kernels and triangle row stamper.

Cloth integrates with the rigid PGS pipeline via the partitioner-driven
graph coloring -- see ``_constraints_to_elements_kernel`` and the
``cloth_support=True`` factory variants of the singleworld iterate /
prepare / relax kernels in ``solver_phoenx_kernels.py``. The kernels
in this file are the bookkeeping pieces that don't fit into the per-cid
type-tag dispatch:

* ``cloth_predict_kernel`` -- per particle: snapshot pose into
  ``position_prev_substep``, set ``access_mode``, apply gravity, predict.
* ``cloth_recover_kernel`` -- per particle: flip access_mode to
  ``VELOCITY_LEVEL`` so the position delta is folded back into velocity.
* ``cloth_init_triangle_rows_kernel`` -- one-shot row populator from a
  Newton :class:`~newton.Model`'s triangle mesh.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_STATIC,
    ACCESS_MODE_VELOCITY_LEVEL,
    synchronize_position_velocity,
)
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
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
    "cloth_predict_kernel",
    "cloth_recover_kernel",
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

    Particles with ``inverse_mass == 0`` get ``access_mode = STATIC``.
    Nonzero velocity advances them kinematically; zero velocity keeps
    ordinary pinned particles anchored.
    """
    i = wp.tid()
    p = particles.position[i]
    particles.position_prev_substep[i] = p
    if particles.inverse_mass[i] == wp.float32(0.0):
        particles.access_mode[i] = ACCESS_MODE_STATIC
        particles.position[i] = p + substep_dt * particles.velocity[i]
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
def cloth_init_triangle_rows_kernel(
    constraints: ConstraintContainer,
    cid_offset: wp.int32,
    num_bodies: wp.int32,
    tri_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3f],
    tri_materials: wp.array2d[wp.float32],
    default_beta_lambda: wp.float32,
    default_beta_mu: wp.float32,
):
    """Stamp one cloth-triangle row from Newton mesh API.

    Body fields are stamped in **unified indexing**: rigid bodies live
    at ``[0, num_bodies)`` and particles at
    ``[num_bodies, num_bodies + num_particles)``. The cloth iterate
    subtracts ``num_bodies`` before indexing the particle SoA; the
    graph-coloring partitioner reads the body fields verbatim and sees
    one element with three unified-index members per cloth triangle.

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
    cloth_triangle_set_body1(constraints, cid, num_bodies + pa)
    cloth_triangle_set_body2(constraints, cid, num_bodies + pb)
    cloth_triangle_set_body3(constraints, cid, num_bodies + pc)

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
                    ac_y * inv_det,
                    -ac_x * inv_det,
                    -ab_y * inv_det,
                    ab_x * inv_det,
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
