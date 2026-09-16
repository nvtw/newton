# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Rigid color-group dispatch with existing copy-aware constraint callbacks."""

import functools

import warp as wp

from newton._src.solvers.phoenx.constraints.bilateral_joint import get_iterate_bilateral_joint_block
from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    BodyContainer,
    ConstraintContainer,
    ContactColumnContainer,
    ContactContainer,
    ContactViews,
    CopyStateContainer,
    ParticleContainer,
    _make_singleworld_dispatch_func,
    _sync_threads,
    joint_constraint_iterate_inequality,
)

# Keep the launch grid and the independent-group stride in sync.
DEFAULT_SWEEP_BLOCK_COUNT = 64
CONSTRAINTS_PER_BLOCK = 32
JOINT_RHS_LANES = 8


def use_cooperative_joint_rhs(*, phase: str, is_cuda: bool, has_bilateral_blocks: bool) -> bool:
    """Keep preparation, contact-only scenes and CPU execution scalar."""
    return is_cuda and has_bilateral_blocks and phase in ("iterate", "relax")


def get_sweep_block_dim(cooperative_joints: bool) -> int:
    """Match the launch width to the compile-time constraint lane mapping."""
    return CONSTRAINTS_PER_BLOCK * (JOINT_RHS_LANES if cooperative_joints else 1)


@functools.cache
def get_sweep_kernel(phase, soft_pd, block_count=DEFAULT_SWEEP_BLOCK_COUNT, *, cooperative_joints=False):
    """Build an ordered sweep within each independent color group."""
    if block_count < 1:
        raise ValueError("Color-group dispatch requires at least one block")
    if cooperative_joints and phase not in ("iterate", "relax"):
        raise ValueError("Cooperative joint RHS is only available for iteration and relaxation")
    lanes_per_constraint = JOINT_RHS_LANES if cooperative_joints else 1
    joint_iterate = get_iterate_bilateral_joint_block(True)
    use_bias = phase == "iterate"
    dispatch, _ = _make_singleworld_dispatch_func(
        cloth_support=False,
        soft_tet_neohookean=False,
        enable_column_timers=False,
        has_joints=True,
        skip_joint_pgs=False,
        has_mass_splitting=True,
        packed_contact_headers=False,
        has_sleeping=False,
        has_soft_contact_pd=soft_pd,
        is_prepare=phase == "prepare",
        is_cached_prepare=phase == "cached_prepare",
        use_bias=phase == "iterate",
        patch_friction=False,
        bilateral_joint_blocks=True,
    )

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
    ):
        block, lane = wp.tid()
        for slab in range(block, (num_colors[0] + slab_width - 1) / slab_width, block_count):
            for local_color in range(slab_width):
                color = slab * slab_width + local_color
                if color < num_colors[0]:
                    for index in range(
                        starts[color] + lane / lanes_per_constraint, starts[color + 1], CONSTRAINTS_PER_BLOCK
                    ):
                        cid = ids[index]
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
                            elif lane % JOINT_RHS_LANES == 0:
                                dispatch(
                                    constraints,
                                    columns,
                                    bodies,
                                    particles,
                                    cc,
                                    contacts,
                                    copies,
                                    num_joints,
                                    enabled,
                                    0,
                                    0,
                                    0,
                                    0,
                                    num_bodies,
                                    idt,
                                    1.0,
                                    cid,
                                    index,
                                    slab,
                                )
                        else:
                            dispatch(
                                constraints,
                                columns,
                                bodies,
                                particles,
                                cc,
                                contacts,
                                copies,
                                num_joints,
                                enabled,
                                0,
                                0,
                                0,
                                0,
                                num_bodies,
                                idt,
                                1.0,
                                cid,
                                index,
                                slab,
                            )
                _sync_threads()

    return sweep
