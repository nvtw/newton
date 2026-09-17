# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Iterate common maximal-coordinate D6 inequality rows."""

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.body import BodyContainer, body_set_access_mode
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer, read_int
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    _OFF_BODY1,
    _OFF_BODY2,
    _ms_load_body_pair,
    _ms_store_body_pair,
)
from newton._src.solvers.phoenx.constraints.d6_inequality import iterate_d6_inequalities
from newton._src.solvers.phoenx.mass_splitting import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer


@wp.func
def joint_constraint_iterate_inequality(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
):
    """Iterate common D6 limit, speed-cap, and friction rows."""
    if constraints.d6.enabled == wp.int32(0) or constraints.d6.row_count[cid] == wp.int32(0):
        return

    body1 = read_int(constraints, _OFF_BODY1, cid)
    body2 = read_int(constraints, _OFF_BODY2, cid)
    body_set_access_mode(bodies, body1, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_set_access_mode(bodies, body2, ACCESS_MODE_VELOCITY_LEVEL, idt)
    (
        velocity1,
        velocity2,
        angular_velocity1,
        angular_velocity2,
        inverse_mass1,
        inverse_mass2,
        inverse_inertia1,
        inverse_inertia2,
        slot1,
        slot2,
    ) = _ms_load_body_pair(
        bodies,
        particles,
        copy_state,
        body1,
        body2,
        parallel_id,
        num_bodies,
    )
    velocity1, angular_velocity1, velocity2, angular_velocity2 = iterate_d6_inequalities(
        constraints.d6,
        cid,
        inverse_mass1,
        inverse_mass2,
        inverse_inertia1,
        inverse_inertia2,
        velocity1,
        angular_velocity1,
        velocity2,
        angular_velocity2,
        idt,
        sor_boost,
    )
    _ms_store_body_pair(
        bodies,
        particles,
        copy_state,
        body1,
        body2,
        slot1,
        slot2,
        num_bodies,
        velocity1,
        angular_velocity1,
        velocity2,
        angular_velocity2,
    )


__all__ = ["joint_constraint_iterate_inequality"]
