# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Temporal rigid color-group sweep with explicit persistent contact state."""

import functools

import warp as wp

from newton._src.solvers.phoenx.constraints.bilateral_joint import get_iterate_bilateral_joint_block
from newton._src.solvers.phoenx.constraints.contact_static_ownership import static_owner
from newton._src.solvers.phoenx.constraints.contact_tgs import ContactTGS
from newton._src.solvers.phoenx.constraints.contact_tgs_dynamic import make_iterate
from newton._src.solvers.phoenx.dispatch.color_groups import (
    CONSTRAINTS_PER_BLOCK,
    DEFAULT_SWEEP_BLOCK_COUNT,
    JOINT_RHS_LANES,
)
from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    BodyContainer,
    ConstraintContainer,
    ContactColumnContainer,
    ContactContainer,
    ContactViews,
    CopyStateContainer,
    ParticleContainer,
    _make_singleworld_rigid_joint_dispatch_func,
    _sync_threads,
    joint_constraint_iterate_inequality,
)


@functools.cache
def get_sweep_kernel(
    phase,
    block_count=DEFAULT_SWEEP_BLOCK_COUNT,
    *,
    cooperative_joints=False,
    temporal_springs=True,
    record_wrenches=False,
):
    """Solve rigid groups after parallel contact preparation; static contacts run separately."""
    if phase not in ("prepare", "cached_prepare", "iterate", "relax") or block_count < 1:
        raise ValueError("Expected a valid temporal sweep phase and positive block count")
    if cooperative_joints and phase not in ("iterate", "relax"):
        raise ValueError("Cooperative joints require iterate or relax")
    lanes_per_constraint = JOINT_RHS_LANES if cooperative_joints else 1
    use_bias = phase == "iterate"
    preparing = phase in ("prepare", "cached_prepare")
    joint_iterate = get_iterate_bilateral_joint_block(True, temporal_springs=temporal_springs)
    joint_dispatch = _make_singleworld_rigid_joint_dispatch_func(
        is_prepare=phase == "prepare",
        is_cached_prepare=phase == "cached_prepare",
        use_bias=use_bias,
        enable_column_timers=False,
        bilateral_joint_blocks=True,
        temporal_springs=temporal_springs,
    )
    contact_iterate = make_iterate(mass_splitting=True, biased=use_bias, record_wrenches=record_wrenches)
    contact_cooperative = make_iterate(
        mass_splitting=True, biased=use_bias, cooperative=True, record_wrenches=record_wrenches
    )

    @wp.func
    def dispatch(
        constraints: ConstraintContainer,
        columns: ContactColumnContainer,
        state: ContactTGS,
        bodies: BodyContainer,
        particles: ParticleContainer,
        cc: ContactContainer,
        copies: CopyStateContainer,
        num_joints: int,
        enabled: wp.array[int],
        num_bodies: int,
        idt: float,
        cid: int,
        slab: int,
    ):
        if cid < num_joints:
            if (wp.static(preparing) and enabled[cid] != 0) or enabled[cid] == 1:
                joint_dispatch(constraints, bodies, particles, copies, num_bodies, idt, 1.0, cid, slab)
        elif wp.static(not preparing):
            contact = cid - num_joints
            if static_owner(columns, contact, bodies) < 0:
                contact_iterate(columns, state, contact, bodies, cc, copies, idt, 0)

    @wp.kernel(enable_backward=False, module="unique")
    def sweep(
        constraints: ConstraintContainer,
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        cc: ContactContainer,
        contacts: ContactViews,
        copies: CopyStateContainer,
        num_joints: wp.int32,
        enabled: wp.array[wp.int32],
        num_bodies: wp.int32,
        idt: wp.float32,
        ids: wp.array[wp.int32],
        starts: wp.array[wp.int32],
        num_colors: wp.array[wp.int32],
        slab_width: wp.int32,
        state: ContactTGS,
    ):
        block, lane = wp.tid()
        for slab in range(block, (num_colors[0] + slab_width - 1) / slab_width, block_count):
            # Springs advance once per substep. Solve contacts before joints
            # within each copy group so later contact rows cannot immediately
            # erase the freshly solved drive velocity before integration.
            # Each row still runs exactly once; color ownership is unchanged.
            for kind in range(wp.static(1 if preparing else 2)):
                for local_color in range(slab_width):
                    color = slab * slab_width + local_color
                    if color < num_colors[0]:
                        for index in range(
                            starts[color] + lane / lanes_per_constraint, starts[color + 1], CONSTRAINTS_PER_BLOCK
                        ):
                            cid = ids[index]
                            if wp.static(not preparing):
                                if (cid < num_joints) == (kind == 0):
                                    continue
                            if wp.static(cooperative_joints):
                                if cid < num_joints:
                                    if enabled[cid] == wp.int32(1):
                                        joint_iterate(
                                            constraints,
                                            cid,
                                            bodies,
                                            particles,
                                            copies,
                                            num_bodies,
                                            slab,
                                            use_bias,
                                            lane % JOINT_RHS_LANES,
                                        )
                                        if lane % JOINT_RHS_LANES == 0:
                                            joint_constraint_iterate_inequality(
                                                constraints,
                                                cid,
                                                bodies,
                                                particles,
                                                copies,
                                                num_bodies,
                                                slab,
                                                idt,
                                                1.0,
                                                use_bias,
                                            )
                                else:
                                    contact = cid - num_joints
                                    if static_owner(columns, contact, bodies) < 0:
                                        contact_cooperative(
                                            columns, state, contact, bodies, cc, copies, idt, lane % JOINT_RHS_LANES
                                        )
                            else:
                                dispatch(
                                    constraints,
                                    columns,
                                    state,
                                    bodies,
                                    particles,
                                    cc,
                                    copies,
                                    num_joints,
                                    enabled,
                                    num_bodies,
                                    idt,
                                    cid,
                                    slab,
                                )
                    _sync_threads()

    return sweep
