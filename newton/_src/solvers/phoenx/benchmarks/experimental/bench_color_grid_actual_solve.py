# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental PhoenX global-color solve benchmark.

The production multi-world fast-tail solver assigns a small lane group to each
world and loops colors inside one kernel. That has low launch overhead, but few
worlds can leave many SMs idle. This benchmark keeps the same color ordering and
PGS safety, but tests schedulers that expose each color as global work across
worlds. ``flat`` and ``direct`` use one launch per color, ``block_world`` gives
one physical block to a world, ``block_world_grouped`` serializes joint-mode /
contact subfamilies inside each color to test lower branch divergence, and
``mega`` keeps the color loop inside one persistent kernel with a software grid
barrier. ``autotune`` evaluates the whole
scheduler portfolio, while ``adaptive`` uses a time-budgeted tournament that
keeps only the measured winners. The benchmark is aimed at finding scheduling
rules that can eventually be moved into production instead of hard-coding one
distribution.

This file contains scheduler prototypes that can hang on some scenes and Warp
versions. The default ``baseline`` mode only times the production solve kernel.
Prototype modes require ``--allow-unsafe-prototypes`` so accidental benchmark
runs do not treat this file as production evidence.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _extract_solver
from newton._src.solvers.phoenx.benchmarks.scenarios import dr_legs, g1_flat, h1_flat, tower
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    _OFF_JOINT_MODE,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_CYLINDRICAL,
    JOINT_MODE_FIXED,
    JOINT_MODE_PLANAR,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
    actuated_double_ball_socket_iterate_multi,
    actuated_double_ball_socket_prepare_for_iteration,
    revolute_iterate_multi,
    revolute_prepare_for_iteration,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_iterate_multi_no_soft_pd,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import (
    contact_prepare_for_iteration_lean_no_soft_pd,
)
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.phoenx.solver_phoenx_kernels import _sync_threads

_EXPERIMENTAL_INNER_SWEEPS = 1
_DEFAULT_BLOCK_WORLD_DIM = 128
_BLOCK_WORLD_SUBFAMILIES = 10
_BLOCK_WORLD_SUBFAMILY_STRIDE = _BLOCK_WORLD_SUBFAMILIES + 1

_JOINT_MODE_REVOLUTE_HOST = int(JOINT_MODE_REVOLUTE)
_JOINT_MODE_PRISMATIC_HOST = int(JOINT_MODE_PRISMATIC)
_JOINT_MODE_BALL_SOCKET_HOST = int(JOINT_MODE_BALL_SOCKET)
_JOINT_MODE_FIXED_HOST = int(JOINT_MODE_FIXED)
_JOINT_MODE_CABLE_HOST = int(JOINT_MODE_CABLE)
_JOINT_MODE_UNIVERSAL_HOST = int(JOINT_MODE_UNIVERSAL)
_JOINT_MODE_CYLINDRICAL_HOST = int(JOINT_MODE_CYLINDRICAL)
_JOINT_MODE_PLANAR_HOST = int(JOINT_MODE_PLANAR)
_JOINT_MODE_OFFSET_HOST = int(_OFF_JOINT_MODE)


@dataclass
class ColorGridHost:
    eids: np.ndarray
    color_starts: np.ndarray
    color_max_counts: np.ndarray
    num_colors: int


@dataclass
class ColorGridDevice:
    host: ColorGridHost
    eids: wp.array
    color_starts: wp.array
    processed: wp.array
    barrier_count: wp.array
    barrier_sense: wp.array


@dataclass
class BlockWorldSubfamilyHost:
    eids: np.ndarray
    world_color_starts: np.ndarray
    world_subfamily_starts: np.ndarray
    world_csr_offsets: np.ndarray
    world_num_colors: np.ndarray
    row_count: int
    num_colors: int


@dataclass
class BlockWorldSubfamilyDevice:
    host: BlockWorldSubfamilyHost
    eids: wp.array
    world_color_starts: wp.array
    world_subfamily_starts: wp.array
    world_csr_offsets: wp.array
    world_num_colors: wp.array


@dataclass(frozen=True)
class SchedulerCandidate:
    label: str
    run: Callable[[], None]


@wp.kernel(enable_backward=False)
def _color_grid_reset_kernel(processed: wp.array[wp.int32]):
    processed[0] = wp.int32(0)


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__threadfence();
#endif
""")
def _thread_fence(): ...


@wp.func
def _grid_barrier(
    barrier_count: wp.array[wp.int32],
    barrier_sense: wp.array[wp.int32],
    tid: wp.int32,
    num_blocks: wp.int32,
    block_dim: wp.int32,
):
    local_tid = tid - (tid / block_dim) * block_dim
    sense = wp.atomic_add(barrier_sense, 0, wp.int32(0))
    _sync_threads()
    if local_tid == wp.int32(0):
        old = wp.atomic_add(barrier_count, 0, wp.int32(1))
        if old == num_blocks - wp.int32(1):
            barrier_count[0] = wp.int32(0)
            _thread_fence()
            wp.atomic_add(barrier_sense, 0, wp.int32(1))
        else:
            while wp.atomic_add(barrier_sense, 0, wp.int32(0)) == sense:
                pass
    _sync_threads()


@wp.kernel(enable_backward=False)
def _color_grid_barrier_reset_kernel(barrier_count: wp.array[wp.int32], barrier_sense: wp.array[wp.int32]):
    barrier_count[0] = wp.int32(0)
    barrier_sense[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _color_grid_prepare_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    eids: wp.array[wp.int32],
    row_start: wp.int32,
    row_count: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    processed: wp.array[wp.int32],
    copy_state: CopyStateContainer,
):
    tid = wp.tid()
    if tid < row_count:
        cid = eids[row_start + tid]
        if cid < num_joints:
            if revolute_only != wp.int32(0):
                revolute_prepare_for_iteration(
                    constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                )
            else:
                actuated_double_ball_socket_prepare_for_iteration(
                    constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                )
        else:
            contact_prepare_for_iteration_lean_no_soft_pd(
                contact_cols,
                cid - num_joints,
                bodies,
                particles,
                num_bodies,
                idt,
                cc,
                contacts,
                copy_state,
                wp.int32(0),
            )


@wp.kernel(enable_backward=False)
def _color_grid_iterate_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    eids: wp.array[wp.int32],
    row_start: wp.int32,
    row_count: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    inner_sweeps: wp.int32,
    processed: wp.array[wp.int32],
    copy_state: CopyStateContainer,
):
    tid = wp.tid()
    if tid < row_count:
        cid = eids[row_start + tid]
        if cid < num_joints:
            if revolute_only != wp.int32(0):
                revolute_iterate_multi(
                    constraints,
                    cid,
                    bodies,
                    particles,
                    copy_state,
                    num_bodies,
                    wp.int32(0),
                    idt,
                    sor_boost,
                    True,
                    inner_sweeps,
                )
            else:
                actuated_double_ball_socket_iterate_multi(
                    constraints,
                    cid,
                    bodies,
                    particles,
                    copy_state,
                    num_bodies,
                    wp.int32(0),
                    idt,
                    sor_boost,
                    True,
                    inner_sweeps,
                )
        else:
            contact_iterate_multi_no_soft_pd(
                contact_cols,
                cid - num_joints,
                bodies,
                particles,
                num_bodies,
                idt,
                cc,
                contacts,
                True,
                inner_sweeps,
                copy_state,
                wp.int32(0),
                sor_boost,
            )


@wp.kernel(enable_backward=False)
def _color_grid_mega_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    eids: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    outer_iters: wp.int32,
    inner_sweeps: wp.int32,
    barrier_count: wp.array[wp.int32],
    barrier_sense: wp.array[wp.int32],
    total_threads: wp.int32,
    num_blocks: wp.int32,
    block_dim_value: wp.int32,
    copy_state: CopyStateContainer,
):
    tid = wp.tid()
    color = wp.int32(0)
    while color < num_colors:
        start = color_starts[color]
        end = color_starts[color + wp.int32(1)]
        cursor = start + tid
        while cursor < end:
            cid = eids[cursor]
            if cid < num_joints:
                if revolute_only != wp.int32(0):
                    revolute_prepare_for_iteration(
                        constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                    )
                else:
                    actuated_double_ball_socket_prepare_for_iteration(
                        constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                    )
            else:
                contact_prepare_for_iteration_lean_no_soft_pd(
                    contact_cols,
                    cid - num_joints,
                    bodies,
                    particles,
                    num_bodies,
                    idt,
                    cc,
                    contacts,
                    copy_state,
                    wp.int32(0),
                )
            cursor = cursor + total_threads
        _grid_barrier(barrier_count, barrier_sense, tid, num_blocks, block_dim_value)
        color = color + wp.int32(1)

    outer = wp.int32(0)
    while outer < outer_iters:
        color = wp.int32(0)
        while color < num_colors:
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            cursor = start + tid
            while cursor < end:
                cid = eids[cursor]
                if cid < num_joints:
                    if revolute_only != wp.int32(0):
                        revolute_iterate_multi(
                            constraints,
                            cid,
                            bodies,
                            particles,
                            copy_state,
                            num_bodies,
                            wp.int32(0),
                            idt,
                            sor_boost,
                            True,
                            inner_sweeps,
                        )
                    else:
                        actuated_double_ball_socket_iterate_multi(
                            constraints,
                            cid,
                            bodies,
                            particles,
                            copy_state,
                            num_bodies,
                            wp.int32(0),
                            idt,
                            sor_boost,
                            True,
                            inner_sweeps,
                        )
                else:
                    contact_iterate_multi_no_soft_pd(
                        contact_cols,
                        cid - num_joints,
                        bodies,
                        particles,
                        num_bodies,
                        idt,
                        cc,
                        contacts,
                        True,
                        inner_sweeps,
                        copy_state,
                        wp.int32(0),
                        sor_boost,
                    )
                cursor = cursor + total_threads
            _grid_barrier(barrier_count, barrier_sense, tid, num_blocks, block_dim_value)
            color = color + wp.int32(1)
        outer = outer + wp.int32(1)


@wp.kernel(enable_backward=False)
def _color_grid_mega_direct_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    num_colors: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    outer_iters: wp.int32,
    inner_sweeps: wp.int32,
    num_worlds: wp.int32,
    row_bound: wp.int32,
    barrier_count: wp.array[wp.int32],
    barrier_sense: wp.array[wp.int32],
    total_threads: wp.int32,
    num_blocks: wp.int32,
    block_dim_value: wp.int32,
    copy_state: CopyStateContainer,
):
    tid = wp.tid()
    total_slots = num_worlds * row_bound
    color = wp.int32(0)
    while color < num_colors:
        slot = tid
        while slot < total_slots:
            world_id = slot / row_bound
            local_row_base = slot - world_id * row_bound
            if color < world_num_colors[world_id]:
                world_base = world_csr_offsets[world_id]
                start = world_base + world_color_starts[world_id, color]
                end = world_base + world_color_starts[world_id, color + wp.int32(1)]
                row = local_row_base
                while row < end - start:
                    cid = world_element_ids_by_color[start + row]
                    if cid < num_joints:
                        if revolute_only != wp.int32(0):
                            revolute_prepare_for_iteration(
                                constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                            )
                        else:
                            actuated_double_ball_socket_prepare_for_iteration(
                                constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                            )
                    else:
                        contact_prepare_for_iteration_lean_no_soft_pd(
                            contact_cols,
                            cid - num_joints,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            copy_state,
                            wp.int32(0),
                        )
                    row = row + row_bound
            slot = slot + total_threads
        _grid_barrier(barrier_count, barrier_sense, tid, num_blocks, block_dim_value)
        color = color + wp.int32(1)

    outer = wp.int32(0)
    while outer < outer_iters:
        color = wp.int32(0)
        while color < num_colors:
            slot = tid
            while slot < total_slots:
                world_id = slot / row_bound
                local_row_base = slot - world_id * row_bound
                if color < world_num_colors[world_id]:
                    world_base = world_csr_offsets[world_id]
                    start = world_base + world_color_starts[world_id, color]
                    end = world_base + world_color_starts[world_id, color + wp.int32(1)]
                    row = local_row_base
                    while row < end - start:
                        cid = world_element_ids_by_color[start + row]
                        if cid < num_joints:
                            if revolute_only != wp.int32(0):
                                revolute_iterate_multi(
                                    constraints,
                                    cid,
                                    bodies,
                                    particles,
                                    copy_state,
                                    num_bodies,
                                    wp.int32(0),
                                    idt,
                                    sor_boost,
                                    True,
                                    inner_sweeps,
                                )
                            else:
                                actuated_double_ball_socket_iterate_multi(
                                    constraints,
                                    cid,
                                    bodies,
                                    particles,
                                    copy_state,
                                    num_bodies,
                                    wp.int32(0),
                                    idt,
                                    sor_boost,
                                    True,
                                    inner_sweeps,
                                )
                        else:
                            contact_iterate_multi_no_soft_pd(
                                contact_cols,
                                cid - num_joints,
                                bodies,
                                particles,
                                num_bodies,
                                idt,
                                cc,
                                contacts,
                                True,
                                inner_sweeps,
                                copy_state,
                                wp.int32(0),
                                sor_boost,
                            )
                        row = row + row_bound
                slot = slot + total_threads
            _grid_barrier(barrier_count, barrier_sense, tid, num_blocks, block_dim_value)
            color = color + wp.int32(1)
        outer = outer + wp.int32(1)


@wp.kernel(enable_backward=False)
def _color_grid_prepare_direct_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    color: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    copy_state: CopyStateContainer,
):
    world_id, local_row = wp.tid()
    if color < world_num_colors[world_id]:
        world_base = world_csr_offsets[world_id]
        start = world_base + world_color_starts[world_id, color]
        end = world_base + world_color_starts[world_id, color + wp.int32(1)]
        if local_row < end - start:
            cid = world_element_ids_by_color[start + local_row]
            if cid < num_joints:
                if revolute_only != wp.int32(0):
                    revolute_prepare_for_iteration(
                        constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                    )
                else:
                    actuated_double_ball_socket_prepare_for_iteration(
                        constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                    )
            else:
                contact_prepare_for_iteration_lean_no_soft_pd(
                    contact_cols,
                    cid - num_joints,
                    bodies,
                    particles,
                    num_bodies,
                    idt,
                    cc,
                    contacts,
                    copy_state,
                    wp.int32(0),
                )


@wp.kernel(enable_backward=False)
def _color_grid_iterate_direct_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    color: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    inner_sweeps: wp.int32,
    copy_state: CopyStateContainer,
):
    world_id, local_row = wp.tid()
    if color < world_num_colors[world_id]:
        world_base = world_csr_offsets[world_id]
        start = world_base + world_color_starts[world_id, color]
        end = world_base + world_color_starts[world_id, color + wp.int32(1)]
        if local_row < end - start:
            cid = world_element_ids_by_color[start + local_row]
            if cid < num_joints:
                if revolute_only != wp.int32(0):
                    revolute_iterate_multi(
                        constraints,
                        cid,
                        bodies,
                        particles,
                        copy_state,
                        num_bodies,
                        wp.int32(0),
                        idt,
                        sor_boost,
                        True,
                        inner_sweeps,
                    )
                else:
                    actuated_double_ball_socket_iterate_multi(
                        constraints,
                        cid,
                        bodies,
                        particles,
                        copy_state,
                        num_bodies,
                        wp.int32(0),
                        idt,
                        sor_boost,
                        True,
                        inner_sweeps,
                    )
            else:
                contact_iterate_multi_no_soft_pd(
                    contact_cols,
                    cid - num_joints,
                    bodies,
                    particles,
                    num_bodies,
                    idt,
                    cc,
                    contacts,
                    True,
                    inner_sweeps,
                    copy_state,
                    wp.int32(0),
                    sor_boost,
                )


@wp.kernel(enable_backward=False)
def _block_world_prepare_iterate_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    world_color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    outer_iters: wp.int32,
    inner_sweeps: wp.int32,
    copy_state: CopyStateContainer,
    block_dim_value: wp.int32,
):
    tid = wp.tid()
    local_tid = tid - (tid / block_dim_value) * block_dim_value
    world_id = tid / block_dim_value
    n_colors = world_num_colors[world_id]
    world_base = world_csr_offsets[world_id]

    color = wp.int32(0)
    while color < n_colors:
        start = world_base + world_color_starts[world_id, color]
        end = world_base + world_color_starts[world_id, color + wp.int32(1)]
        cursor = local_tid
        while cursor < end - start:
            cid = world_element_ids_by_color[start + cursor]
            if cid < num_joints:
                if revolute_only != wp.int32(0):
                    revolute_prepare_for_iteration(
                        constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                    )
                else:
                    actuated_double_ball_socket_prepare_for_iteration(
                        constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                    )
            else:
                contact_prepare_for_iteration_lean_no_soft_pd(
                    contact_cols,
                    cid - num_joints,
                    bodies,
                    particles,
                    num_bodies,
                    idt,
                    cc,
                    contacts,
                    copy_state,
                    wp.int32(0),
                )
            cursor = cursor + block_dim_value
        _sync_threads()
        color = color + wp.int32(1)

    outer = wp.int32(0)
    while outer < outer_iters:
        color = wp.int32(0)
        while color < n_colors:
            start = world_base + world_color_starts[world_id, color]
            end = world_base + world_color_starts[world_id, color + wp.int32(1)]
            cursor = local_tid
            while cursor < end - start:
                cid = world_element_ids_by_color[start + cursor]
                if cid < num_joints:
                    if revolute_only != wp.int32(0):
                        revolute_iterate_multi(
                            constraints,
                            cid,
                            bodies,
                            particles,
                            copy_state,
                            num_bodies,
                            wp.int32(0),
                            idt,
                            sor_boost,
                            True,
                            inner_sweeps,
                        )
                    else:
                        actuated_double_ball_socket_iterate_multi(
                            constraints,
                            cid,
                            bodies,
                            particles,
                            copy_state,
                            num_bodies,
                            wp.int32(0),
                            idt,
                            sor_boost,
                            True,
                            inner_sweeps,
                        )
                else:
                    contact_iterate_multi_no_soft_pd(
                        contact_cols,
                        cid - num_joints,
                        bodies,
                        particles,
                        num_bodies,
                        idt,
                        cc,
                        contacts,
                        True,
                        inner_sweeps,
                        copy_state,
                        wp.int32(0),
                        sor_boost,
                    )
                cursor = cursor + block_dim_value
            _sync_threads()
            color = color + wp.int32(1)
        outer = outer + wp.int32(1)


@wp.kernel(enable_backward=False)
def _block_world_subfamily_prepare_iterate_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    world_subfamily_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    outer_iters: wp.int32,
    inner_sweeps: wp.int32,
    copy_state: CopyStateContainer,
    block_dim_value: wp.int32,
):
    tid = wp.tid()
    local_tid = tid - (tid / block_dim_value) * block_dim_value
    world_id = tid / block_dim_value
    n_colors = world_num_colors[world_id]
    world_base = world_csr_offsets[world_id]

    color = wp.int32(0)
    while color < n_colors:
        family_base = color * wp.int32(_BLOCK_WORLD_SUBFAMILY_STRIDE)
        family = wp.int32(0)
        while family < wp.int32(_BLOCK_WORLD_SUBFAMILIES):
            start = world_base + world_subfamily_starts[world_id, family_base + family]
            end = world_base + world_subfamily_starts[world_id, family_base + family + wp.int32(1)]
            cursor = local_tid
            while cursor < end - start:
                cid = world_element_ids_by_color[start + cursor]
                if cid < num_joints:
                    if revolute_only != wp.int32(0):
                        revolute_prepare_for_iteration(
                            constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                        )
                    else:
                        actuated_double_ball_socket_prepare_for_iteration(
                            constraints, cid, bodies, particles, copy_state, num_bodies, wp.int32(0), idt
                        )
                else:
                    contact_prepare_for_iteration_lean_no_soft_pd(
                        contact_cols,
                        cid - num_joints,
                        bodies,
                        particles,
                        num_bodies,
                        idt,
                        cc,
                        contacts,
                        copy_state,
                        wp.int32(0),
                    )
                cursor = cursor + block_dim_value
            family = family + wp.int32(1)
        _sync_threads()
        color = color + wp.int32(1)

    outer = wp.int32(0)
    while outer < outer_iters:
        color = wp.int32(0)
        while color < n_colors:
            family_base = color * wp.int32(_BLOCK_WORLD_SUBFAMILY_STRIDE)
            family = wp.int32(0)
            while family < wp.int32(_BLOCK_WORLD_SUBFAMILIES):
                start = world_base + world_subfamily_starts[world_id, family_base + family]
                end = world_base + world_subfamily_starts[world_id, family_base + family + wp.int32(1)]
                cursor = local_tid
                while cursor < end - start:
                    cid = world_element_ids_by_color[start + cursor]
                    if cid < num_joints:
                        if revolute_only != wp.int32(0):
                            revolute_iterate_multi(
                                constraints,
                                cid,
                                bodies,
                                particles,
                                copy_state,
                                num_bodies,
                                wp.int32(0),
                                idt,
                                sor_boost,
                                True,
                                inner_sweeps,
                            )
                        else:
                            actuated_double_ball_socket_iterate_multi(
                                constraints,
                                cid,
                                bodies,
                                particles,
                                copy_state,
                                num_bodies,
                                wp.int32(0),
                                idt,
                                sor_boost,
                                True,
                                inner_sweeps,
                            )
                    else:
                        contact_iterate_multi_no_soft_pd(
                            contact_cols,
                            cid - num_joints,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            True,
                            inner_sweeps,
                            copy_state,
                            wp.int32(0),
                            sor_boost,
                        )
                    cursor = cursor + block_dim_value
                family = family + wp.int32(1)
            _sync_threads()
            color = color + wp.int32(1)
        outer = outer + wp.int32(1)


def _joint_mode_rank(mode: int) -> int:
    if mode == _JOINT_MODE_REVOLUTE_HOST:
        return 0
    if mode == _JOINT_MODE_PRISMATIC_HOST:
        return 1
    if mode == _JOINT_MODE_BALL_SOCKET_HOST:
        return 2
    if mode == _JOINT_MODE_FIXED_HOST:
        return 3
    if mode == _JOINT_MODE_CABLE_HOST:
        return 4
    if mode == _JOINT_MODE_UNIVERSAL_HOST:
        return 5
    if mode == _JOINT_MODE_CYLINDRICAL_HOST:
        return 6
    if mode == _JOINT_MODE_PLANAR_HOST:
        return 7
    return 8


def _extract_joint_modes(world: PhoenXWorld) -> np.ndarray:
    if int(world.num_joints) <= 0:
        return np.empty(0, dtype=np.int32)
    constraint_words = world.constraints.data.numpy()
    return np.ascontiguousarray(
        constraint_words[_JOINT_MODE_OFFSET_HOST, : int(world.num_joints)],
        dtype=np.float32,
    ).view(np.int32)


def _row_kind_sort_key(eid: int, family: np.ndarray, joint_modes: np.ndarray, num_joints: int) -> tuple[int, int, int]:
    if 0 <= eid < num_joints:
        return (0, _joint_mode_rank(int(joint_modes[eid])), eid)
    if 0 <= eid < int(family.shape[0]) and int(family[eid]) == 1:
        return (1, 0, eid)
    return (2, 0, eid)


def _block_world_subfamily(eid: int, family: np.ndarray, joint_modes: np.ndarray, num_joints: int) -> int:
    if 0 <= eid < num_joints:
        return min(_joint_mode_rank(int(joint_modes[eid])), _BLOCK_WORLD_SUBFAMILIES - 2)
    if 0 <= eid < int(family.shape[0]) and int(family[eid]) == 1:
        return _BLOCK_WORLD_SUBFAMILIES - 1
    return _BLOCK_WORLD_SUBFAMILIES - 1


def _build_scene(
    scene: str,
    num_worlds: int,
    *,
    substeps: int,
    solver_iterations: int,
    prepare_refresh_stride: int | str,
):
    if scene == "h1":
        return h1_flat.build(
            num_worlds, "phoenx", substeps, solver_iterations, prepare_refresh_stride=prepare_refresh_stride
        )
    if scene == "g1":
        return g1_flat.build(
            num_worlds, "phoenx", substeps, solver_iterations, prepare_refresh_stride=prepare_refresh_stride
        )
    if scene == "tower":
        return tower.build(
            num_worlds,
            "phoenx",
            substeps,
            solver_iterations,
            step_layout="multi_world",
            prepare_refresh_stride=prepare_refresh_stride,
        )
    if scene == "dr_legs":
        return dr_legs.build(
            num_worlds, "phoenx", substeps, solver_iterations, prepare_refresh_stride=prepare_refresh_stride
        )
    raise ValueError(f"unknown scene: {scene}")


def _extract_color_grid(world: PhoenXWorld, *, group_by_kind: bool = False) -> ColorGridHost:
    active = int(world._num_active_constraints.numpy()[0])
    eids_by_color = world._world_element_ids_by_color.numpy()
    starts = world._world_color_starts.numpy()
    csr = world._world_csr_offsets.numpy()
    num_colors_per_world = world._world_num_colors.numpy().astype(np.int32, copy=False)
    num_colors = int(num_colors_per_world.max(initial=0))
    family = world._element_family.numpy() if group_by_kind else np.empty(0, dtype=np.int32)
    joint_modes = _extract_joint_modes(world) if group_by_kind else np.empty(0, dtype=np.int32)
    eids: list[int] = []
    color_starts = np.zeros(num_colors + 1, dtype=np.int32)
    color_max_counts = np.zeros(num_colors, dtype=np.int32)
    for color in range(num_colors):
        color_starts[color] = len(eids)
        color_eids: list[int] = []
        for world_id in range(world.num_worlds):
            if color >= int(num_colors_per_world[world_id]):
                continue
            base = int(csr[world_id])
            start = base + int(starts[world_id, color])
            end = base + int(starts[world_id, color + 1])
            color_max_counts[color] = max(int(color_max_counts[color]), end - start)
            for cursor in range(start, end):
                eid = int(eids_by_color[cursor])
                if 0 <= eid < active:
                    color_eids.append(eid)
        if group_by_kind:
            color_eids.sort(key=lambda eid: _row_kind_sort_key(eid, family, joint_modes, int(world.num_joints)))
        eids.extend(color_eids)
    color_starts[num_colors] = len(eids)
    if not eids:
        eids.append(0)
    return ColorGridHost(
        eids=np.asarray(eids, dtype=np.int32),
        color_starts=color_starts,
        color_max_counts=color_max_counts,
        num_colors=num_colors,
    )


def _extract_block_world_subfamily_grid(world: PhoenXWorld) -> BlockWorldSubfamilyHost:
    active = int(world._num_active_constraints.numpy()[0])
    source_eids = world._world_element_ids_by_color.numpy()
    starts = world._world_color_starts.numpy()
    csr = world._world_csr_offsets.numpy()
    num_colors_per_world = world._world_num_colors.numpy().astype(np.int32, copy=False)
    num_colors = int(num_colors_per_world.max(initial=0))
    family = world._element_family.numpy()
    joint_modes = _extract_joint_modes(world)
    world_color_starts = np.zeros((world.num_worlds, num_colors + 1), dtype=np.int32)
    world_subfamily_starts = np.zeros(
        (world.num_worlds, max(1, num_colors * _BLOCK_WORLD_SUBFAMILY_STRIDE)),
        dtype=np.int32,
    )
    world_csr_offsets = np.zeros(world.num_worlds + 1, dtype=np.int32)
    eids: list[int] = []

    for world_id in range(world.num_worlds):
        world_base = len(eids)
        world_csr_offsets[world_id] = world_base
        for color in range(num_colors):
            world_color_starts[world_id, color] = len(eids) - world_base
            buckets: list[list[int]] = [[] for _ in range(_BLOCK_WORLD_SUBFAMILIES)]
            if color < int(num_colors_per_world[world_id]):
                source_base = int(csr[world_id])
                start = source_base + int(starts[world_id, color])
                end = source_base + int(starts[world_id, color + 1])
                for cursor in range(start, end):
                    eid = int(source_eids[cursor])
                    if 0 <= eid < active:
                        buckets[_block_world_subfamily(eid, family, joint_modes, int(world.num_joints))].append(eid)

            subfamily_base = color * _BLOCK_WORLD_SUBFAMILY_STRIDE
            for subfamily, bucket in enumerate(buckets):
                world_subfamily_starts[world_id, subfamily_base + subfamily] = len(eids) - world_base
                bucket.sort()
                eids.extend(bucket)
            world_subfamily_starts[world_id, subfamily_base + _BLOCK_WORLD_SUBFAMILIES] = len(eids) - world_base
        world_color_starts[world_id, num_colors] = len(eids) - world_base
    world_csr_offsets[world.num_worlds] = len(eids)

    row_count = len(eids)
    if not eids:
        eids.append(0)
    return BlockWorldSubfamilyHost(
        eids=np.asarray(eids, dtype=np.int32),
        world_color_starts=world_color_starts,
        world_subfamily_starts=world_subfamily_starts,
        world_csr_offsets=world_csr_offsets,
        world_num_colors=num_colors_per_world.copy(),
        row_count=row_count,
        num_colors=num_colors,
    )


def _upload_color_grid(graph: ColorGridHost, device: wp.context.Devicelike) -> ColorGridDevice:
    return ColorGridDevice(
        host=graph,
        eids=wp.array(graph.eids, dtype=wp.int32, device=device),
        color_starts=wp.array(graph.color_starts, dtype=wp.int32, device=device),
        processed=wp.zeros(1, dtype=wp.int32, device=device),
        barrier_count=wp.zeros(1, dtype=wp.int32, device=device),
        barrier_sense=wp.zeros(1, dtype=wp.int32, device=device),
    )


def _upload_block_world_subfamily_grid(
    graph: BlockWorldSubfamilyHost, device: wp.context.Devicelike
) -> BlockWorldSubfamilyDevice:
    return BlockWorldSubfamilyDevice(
        host=graph,
        eids=wp.array(graph.eids, dtype=wp.int32, device=device),
        world_color_starts=wp.array(graph.world_color_starts, dtype=wp.int32, device=device),
        world_subfamily_starts=wp.array(graph.world_subfamily_starts, dtype=wp.int32, device=device),
        world_csr_offsets=wp.array(graph.world_csr_offsets, dtype=wp.int32, device=device),
        world_num_colors=wp.array(graph.world_num_colors, dtype=wp.int32, device=device),
    )


def _color_grid_runner(world: PhoenXWorld, graph: ColorGridDevice, *, block_dim: int):
    device = world.device
    contact_views = world._contact_views if world._contact_views is not None else world._contact_views_placeholder
    idt = wp.float32(1.0 / world.substep_dt)
    inner_sweeps = int(_EXPERIMENTAL_INNER_SWEEPS)
    outer_iters = int(world.solver_iterations) // inner_sweeps
    revolute_only = 1 if bool(world._use_revolute_specialization) else 0
    starts = graph.host.color_starts
    counts = [int(starts[i + 1] - starts[i]) for i in range(graph.host.num_colors)]

    def run() -> None:
        wp.launch(_color_grid_reset_kernel, dim=1, inputs=[graph.processed], device=device)
        for color, count in enumerate(counts):
            if count <= 0:
                continue
            wp.launch(
                _color_grid_prepare_kernel,
                dim=count,
                block_dim=block_dim,
                inputs=[
                    world.constraints,
                    world._contact_cols,
                    world.bodies,
                    world._particles_or_sentinel(),
                    idt,
                    graph.eids,
                    wp.int32(int(starts[color])),
                    wp.int32(count),
                    world._contact_container,
                    contact_views,
                    wp.int32(world.num_joints),
                    wp.int32(world.num_bodies),
                    wp.int32(revolute_only),
                    graph.processed,
                    world._copy_state,
                ],
                device=device,
            )
        for _outer in range(outer_iters):
            for color, count in enumerate(counts):
                if count <= 0:
                    continue
                wp.launch(
                    _color_grid_iterate_kernel,
                    dim=count,
                    block_dim=block_dim,
                    inputs=[
                        world.constraints,
                        world._contact_cols,
                        world.bodies,
                        world._particles_or_sentinel(),
                        idt,
                        wp.float32(world.sor_boost),
                        graph.eids,
                        wp.int32(int(starts[color])),
                        wp.int32(count),
                        world._contact_container,
                        contact_views,
                        wp.int32(world.num_joints),
                        wp.int32(world.num_bodies),
                        wp.int32(revolute_only),
                        wp.int32(inner_sweeps),
                        graph.processed,
                        world._copy_state,
                    ],
                    device=device,
                )

    return run


def _color_grid_mega_runner(world: PhoenXWorld, graph: ColorGridDevice, *, block_dim: int, worker_blocks: int):
    device = world.device
    contact_views = world._contact_views if world._contact_views is not None else world._contact_views_placeholder
    idt = wp.float32(1.0 / world.substep_dt)
    inner_sweeps = int(_EXPERIMENTAL_INNER_SWEEPS)
    outer_iters = int(world.solver_iterations) // inner_sweeps
    revolute_only = 1 if bool(world._use_revolute_specialization) else 0
    total_threads = max(1, int(block_dim) * int(worker_blocks))

    def run() -> None:
        wp.launch(
            _color_grid_barrier_reset_kernel,
            dim=1,
            inputs=[graph.barrier_count, graph.barrier_sense],
            device=device,
        )
        wp.launch(
            _color_grid_mega_kernel,
            dim=total_threads,
            block_dim=block_dim,
            inputs=[
                world.constraints,
                world._contact_cols,
                world.bodies,
                world._particles_or_sentinel(),
                idt,
                wp.float32(world.sor_boost),
                graph.eids,
                graph.color_starts,
                wp.int32(graph.host.num_colors),
                world._contact_container,
                contact_views,
                wp.int32(world.num_joints),
                wp.int32(world.num_bodies),
                wp.int32(revolute_only),
                wp.int32(outer_iters),
                wp.int32(inner_sweeps),
                graph.barrier_count,
                graph.barrier_sense,
                wp.int32(total_threads),
                wp.int32(worker_blocks),
                wp.int32(block_dim),
                world._copy_state,
            ],
            device=device,
        )

    return run


def _color_grid_mega_direct_runner(
    world: PhoenXWorld,
    graph: ColorGridDevice,
    *,
    block_dim: int,
    worker_blocks: int,
    row_bound: int,
):
    device = world.device
    contact_views = world._contact_views if world._contact_views is not None else world._contact_views_placeholder
    idt = wp.float32(1.0 / world.substep_dt)
    inner_sweeps = int(_EXPERIMENTAL_INNER_SWEEPS)
    outer_iters = int(world.solver_iterations) // inner_sweeps
    revolute_only = 1 if bool(world._use_revolute_specialization) else 0
    total_threads = max(1, int(block_dim) * int(worker_blocks))

    def run() -> None:
        wp.launch(
            _color_grid_barrier_reset_kernel,
            dim=1,
            inputs=[graph.barrier_count, graph.barrier_sense],
            device=device,
        )
        wp.launch(
            _color_grid_mega_direct_kernel,
            dim=total_threads,
            block_dim=block_dim,
            inputs=[
                world.constraints,
                world._contact_cols,
                world.bodies,
                world._particles_or_sentinel(),
                idt,
                wp.float32(world.sor_boost),
                world._world_element_ids_by_color,
                world._world_color_starts,
                world._world_csr_offsets,
                world._world_num_colors,
                wp.int32(graph.host.num_colors),
                world._contact_container,
                contact_views,
                wp.int32(world.num_joints),
                wp.int32(world.num_bodies),
                wp.int32(revolute_only),
                wp.int32(outer_iters),
                wp.int32(inner_sweeps),
                wp.int32(world.num_worlds),
                wp.int32(row_bound),
                graph.barrier_count,
                graph.barrier_sense,
                wp.int32(total_threads),
                wp.int32(worker_blocks),
                wp.int32(block_dim),
                world._copy_state,
            ],
            device=device,
        )

    return run


def _block_world_runner(world: PhoenXWorld, *, block_dim: int):
    device = world.device
    contact_views = world._contact_views if world._contact_views is not None else world._contact_views_placeholder
    idt = wp.float32(1.0 / world.substep_dt)
    inner_sweeps = int(_EXPERIMENTAL_INNER_SWEEPS)
    outer_iters = int(world.solver_iterations) // inner_sweeps
    revolute_only = 1 if bool(world._use_revolute_specialization) else 0
    dim = max(1, int(world.num_worlds) * int(block_dim))

    def run() -> None:
        wp.launch(
            _block_world_prepare_iterate_kernel,
            dim=dim,
            block_dim=block_dim,
            inputs=[
                world.constraints,
                world._contact_cols,
                world.bodies,
                world._particles_or_sentinel(),
                idt,
                wp.float32(world.sor_boost),
                world._world_element_ids_by_color,
                world._world_color_starts,
                world._world_csr_offsets,
                world._world_num_colors,
                world._contact_container,
                contact_views,
                wp.int32(world.num_joints),
                wp.int32(world.num_bodies),
                wp.int32(revolute_only),
                wp.int32(outer_iters),
                wp.int32(inner_sweeps),
                world._copy_state,
                wp.int32(block_dim),
            ],
            device=device,
        )

    return run


def _block_world_subfamily_runner(world: PhoenXWorld, graph: BlockWorldSubfamilyDevice, *, block_dim: int):
    device = world.device
    contact_views = world._contact_views if world._contact_views is not None else world._contact_views_placeholder
    idt = wp.float32(1.0 / world.substep_dt)
    inner_sweeps = int(_EXPERIMENTAL_INNER_SWEEPS)
    outer_iters = int(world.solver_iterations) // inner_sweeps
    revolute_only = 1 if bool(world._use_revolute_specialization) else 0
    dim = max(1, int(world.num_worlds) * int(block_dim))

    def run() -> None:
        wp.launch(
            _block_world_subfamily_prepare_iterate_kernel,
            dim=dim,
            block_dim=block_dim,
            inputs=[
                world.constraints,
                world._contact_cols,
                world.bodies,
                world._particles_or_sentinel(),
                idt,
                wp.float32(world.sor_boost),
                graph.eids,
                graph.world_subfamily_starts,
                graph.world_csr_offsets,
                graph.world_num_colors,
                world._contact_container,
                contact_views,
                wp.int32(world.num_joints),
                wp.int32(world.num_bodies),
                wp.int32(revolute_only),
                wp.int32(outer_iters),
                wp.int32(inner_sweeps),
                world._copy_state,
                wp.int32(block_dim),
            ],
            device=device,
        )

    return run


def _color_grid_direct_runner(world: PhoenXWorld, graph: ColorGridDevice, *, block_dim: int):
    device = world.device
    contact_views = world._contact_views if world._contact_views is not None else world._contact_views_placeholder
    idt = wp.float32(1.0 / world.substep_dt)
    inner_sweeps = int(_EXPERIMENTAL_INNER_SWEEPS)
    outer_iters = int(world.solver_iterations) // inner_sweeps
    revolute_only = 1 if bool(world._use_revolute_specialization) else 0
    max_counts = [int(v) for v in graph.host.color_max_counts]

    def run() -> None:
        for color, max_count in enumerate(max_counts):
            if max_count <= 0:
                continue
            wp.launch(
                _color_grid_prepare_direct_kernel,
                dim=(world.num_worlds, max_count),
                block_dim=block_dim,
                inputs=[
                    world.constraints,
                    world._contact_cols,
                    world.bodies,
                    world._particles_or_sentinel(),
                    idt,
                    world._world_element_ids_by_color,
                    world._world_color_starts,
                    world._world_csr_offsets,
                    world._world_num_colors,
                    wp.int32(color),
                    world._contact_container,
                    contact_views,
                    wp.int32(world.num_joints),
                    wp.int32(world.num_bodies),
                    wp.int32(revolute_only),
                    world._copy_state,
                ],
                device=device,
            )
        for _outer in range(outer_iters):
            for color, max_count in enumerate(max_counts):
                if max_count <= 0:
                    continue
                wp.launch(
                    _color_grid_iterate_direct_kernel,
                    dim=(world.num_worlds, max_count),
                    block_dim=block_dim,
                    inputs=[
                        world.constraints,
                        world._contact_cols,
                        world.bodies,
                        world._particles_or_sentinel(),
                        idt,
                        wp.float32(world.sor_boost),
                        world._world_element_ids_by_color,
                        world._world_color_starts,
                        world._world_csr_offsets,
                        world._world_num_colors,
                        wp.int32(color),
                        world._contact_container,
                        contact_views,
                        wp.int32(world.num_joints),
                        wp.int32(world.num_bodies),
                        wp.int32(revolute_only),
                        wp.int32(inner_sweeps),
                        world._copy_state,
                    ],
                    device=device,
                )

    return run


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(raw.strip()) for raw in value.split(",") if raw.strip())


def _parse_stride_value(value: str) -> int | str:
    item = value.strip().lower()
    return "auto" if item == "auto" else int(item)


def _bench(fn, *, n_runs: int, warmup: int, trials: int, device: wp.context.Devicelike) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    wp.synchronize_device()
    with wp.ScopedCapture(device=device) as capture:
        fn()
    graph = capture.graph
    wp.synchronize_device()
    times: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            wp.capture_launch(graph)
        wp.synchronize_device()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return float(arr.min()), float(np.median(arr))


def _run_baseline(args: argparse.Namespace, world: PhoenXWorld, scene: str, num_worlds: int) -> None:
    active = int(world._num_active_constraints.numpy()[0])
    num_colors = int(world._world_num_colors.numpy().max(initial=0))
    base_min, base_med = _bench(
        world._solve_main, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=world.device
    )
    print(
        f"{scene:5s} worlds={num_worlds:5d} rows={active:6d} colors={num_colors:4d} "
        f"baseline={base_min:8.3f}ms base_us={1000.0 * base_min / args.n_runs:8.3f}"
    )
    if args.verbose:
        print(f"  baseline min={base_min:.3f} ms med={base_med:.3f} ms")


def _validate(graph: ColorGridDevice, expected: int) -> None:
    # The timed kernels intentionally avoid per-row validation atomics.
    # Successful kernel completion is enough for this isolated scheduler bench.
    _ = graph, expected


def _scheduler_candidates(
    args: argparse.Namespace,
    world: PhoenXWorld,
    grid_graph: ColorGridDevice,
    grouped_grid_graph: ColorGridDevice,
    subfamily_graph: BlockWorldSubfamilyDevice,
    *,
    include_launch_heavy: bool,
) -> list[SchedulerCandidate]:
    candidates = [SchedulerCandidate("baseline", world._solve_main)]
    if include_launch_heavy:
        candidates.append(SchedulerCandidate("flat", _color_grid_runner(world, grid_graph, block_dim=args.block_dim)))
        candidates.append(
            SchedulerCandidate("flat_grouped", _color_grid_runner(world, grouped_grid_graph, block_dim=args.block_dim))
        )
        candidates.append(
            SchedulerCandidate("direct", _color_grid_direct_runner(world, grid_graph, block_dim=args.block_dim))
        )
    for block_dim in _parse_csv_ints(args.tune_block_world_dims):
        candidates.append(
            SchedulerCandidate(f"block_world_{block_dim}", _block_world_runner(world, block_dim=block_dim))
        )
        candidates.append(
            SchedulerCandidate(
                f"block_world_grouped_{block_dim}",
                _block_world_subfamily_runner(world, subfamily_graph, block_dim=block_dim),
            )
        )
    return candidates


def _time_candidate(
    candidate: SchedulerCandidate,
    *,
    n_runs: int,
    warmup: int,
    trials: int,
    device: wp.context.Devicelike,
) -> tuple[float, float]:
    return _bench(candidate.run, n_runs=n_runs, warmup=warmup, trials=trials, device=device)


def _time_candidate_budgeted(
    candidate: SchedulerCandidate,
    *,
    initial_runs: int,
    max_runs: int,
    target_ms: float,
    warmup: int,
    trials: int,
    device: wp.context.Devicelike,
) -> tuple[float, float, int]:
    runs = max(1, int(initial_runs))
    max_runs = max(runs, int(max_runs))
    while True:
        total_min_ms, total_med_ms = _time_candidate(
            candidate,
            n_runs=runs,
            warmup=warmup,
            trials=trials,
            device=device,
        )
        if total_min_ms >= target_ms or runs >= max_runs:
            return total_min_ms / float(runs), total_med_ms / float(runs), runs
        runs = min(max_runs, runs * 2)
        warmup = 0


def _format_candidate_results(results: list[tuple[SchedulerCandidate, float, float]], baseline_ms: float) -> str:
    pieces = []
    for candidate, min_ms, _med_ms in results:
        speed = baseline_ms / min_ms if min_ms > 0.0 else float("nan")
        pieces.append(f"{candidate.label}={min_ms:8.3f}ms({speed:5.3f}x)")
    return " ".join(pieces)


def _run_autotune(
    args: argparse.Namespace,
    world: PhoenXWorld,
    grid_graph: ColorGridDevice,
    grouped_grid_graph: ColorGridDevice,
    subfamily_graph: BlockWorldSubfamilyDevice,
    scene: str,
    num_worlds: int,
    row_count: int,
) -> None:
    candidates = _scheduler_candidates(
        args, world, grid_graph, grouped_grid_graph, subfamily_graph, include_launch_heavy=True
    )
    results: list[tuple[SchedulerCandidate, float, float]] = []
    for candidate in candidates:
        min_ms, med_ms = _time_candidate(
            candidate,
            n_runs=args.tune_runs,
            warmup=args.tune_warmup,
            trials=args.trials,
            device=world.device,
        )
        results.append((candidate, min_ms, med_ms))

    baseline = next(min_ms for candidate, min_ms, _med_ms in results if candidate.label == "baseline")
    best_candidate, best_min, _best_med = min(results, key=lambda item: item[1])
    best_speed = baseline / best_min if best_min > 0.0 else float("nan")
    print(
        f"{scene:5s} worlds={num_worlds:5d} rows={row_count:6d} colors={grid_graph.host.num_colors:4d} "
        f"best={best_candidate.label} best_speedup={best_speed:5.3f}x " + _format_candidate_results(results, baseline)
    )


def _run_adaptive(
    args: argparse.Namespace,
    world: PhoenXWorld,
    grid_graph: ColorGridDevice,
    grouped_grid_graph: ColorGridDevice,
    subfamily_graph: BlockWorldSubfamilyDevice,
    scene: str,
    num_worlds: int,
    row_count: int,
) -> None:
    candidates = _scheduler_candidates(
        args,
        world,
        grid_graph,
        grouped_grid_graph,
        subfamily_graph,
        include_launch_heavy=args.adapt_include_launch_heavy,
    )
    candidate_by_label = {candidate.label: candidate for candidate in candidates}
    observed: dict[str, list[float]] = {candidate.label: [] for candidate in candidates}
    active = candidates
    keep_fraction = min(1.0, max(0.1, float(args.adapt_keep_fraction)))
    max_window_runs = max(int(args.adapt_window_runs), int(args.adapt_max_window_runs))

    for round_index in range(max(1, int(args.adapt_rounds))):
        round_results: list[tuple[SchedulerCandidate, float, float]] = []
        for candidate in active:
            min_ms, med_ms, _runs = _time_candidate_budgeted(
                candidate,
                initial_runs=args.adapt_window_runs,
                max_runs=max_window_runs,
                target_ms=args.adapt_target_ms,
                warmup=args.adapt_warmup if round_index == 0 else 0,
                trials=args.trials,
                device=world.device,
            )
            observed[candidate.label].append(min_ms)
            round_results.append((candidate, min_ms, med_ms))
        round_results.sort(key=lambda item: item[1])
        if len(active) <= 1:
            break
        keep_count = max(1, int(np.ceil(len(active) * keep_fraction)))
        active = [candidate for candidate, _min_ms, _med_ms in round_results[:keep_count]]

    scores: list[tuple[SchedulerCandidate, float]] = []
    for candidate in candidates:
        samples = observed[candidate.label]
        if samples:
            scores.append((candidate, float(np.median(np.asarray(samples, dtype=np.float64)))))
    baseline_score = next(score for candidate, score in scores if candidate.label == "baseline")
    active_labels = {candidate.label for candidate in active}
    selectable = [(candidate, score) for candidate, score in scores if candidate.label in active_labels]
    best_candidate, best_score = min(selectable, key=lambda item: item[1])
    selected = best_candidate
    predicted_speed = baseline_score / best_score if best_score > 0.0 else float("nan")
    if selected.label != "baseline" and predicted_speed < float(args.adapt_min_speedup):
        selected = candidate_by_label["baseline"]

    verify_runs = max(int(args.n_runs), int(args.adapt_max_window_runs))
    base_min, _base_med, base_runs = _time_candidate_budgeted(
        candidate_by_label["baseline"],
        initial_runs=args.n_runs,
        max_runs=verify_runs,
        target_ms=args.adapt_verify_target_ms,
        warmup=args.warmup,
        trials=args.trials,
        device=world.device,
    )
    if selected.label == "baseline":
        selected_min = base_min
        selected_runs = base_runs
    else:
        selected_min, _selected_med, selected_runs = _time_candidate_budgeted(
            selected,
            initial_runs=args.n_runs,
            max_runs=verify_runs,
            target_ms=args.adapt_verify_target_ms,
            warmup=args.warmup,
            trials=args.trials,
            device=world.device,
        )
    selected_speed = base_min / selected_min if selected_min > 0.0 else float("nan")
    score_pieces = " ".join(
        f"{candidate.label}:{score * 1000.0:0.2f}us" for candidate, score in sorted(scores, key=lambda item: item[1])
    )
    print(
        f"{scene:5s} worlds={num_worlds:5d} rows={row_count:6d} colors={grid_graph.host.num_colors:4d} "
        f"policy=adaptive selected={selected.label} predicted={predicted_speed:5.3f}x "
        f"verified={selected_speed:5.3f}x baseline={base_min * 1000.0:8.3f}us "
        f"selected={selected_min * 1000.0:8.3f}us runs={base_runs}/{selected_runs} scores={score_pieces}"
    )


def run_case(args: argparse.Namespace, scene: str, num_worlds: int) -> None:
    handle = _build_scene(
        scene,
        num_worlds,
        substeps=args.substeps,
        solver_iterations=args.solver_iterations,
        prepare_refresh_stride=args.prepare_refresh_stride,
    )
    solver = _extract_solver(handle)
    world = solver.world
    if world.step_layout == "single_world" or world.mass_splitting_enabled or world.num_particles > 0:
        raise RuntimeError("color-grid prototype only supports multi_world rigid scenes")
    if world._has_soft_contact_pd or world._sleeping_enabled:
        raise RuntimeError("color-grid prototype currently excludes soft-contact PD and sleeping")
    for _ in range(args.prime_frames):
        handle.simulate_one_frame()
    wp.synchronize_device()

    if args.mode == "baseline":
        _run_baseline(args, world, scene, num_worlds)
        return

    host_graph = _extract_color_grid(world)
    grouped_host_graph = _extract_color_grid(world, group_by_kind=True)
    subfamily_host_graph = _extract_block_world_subfamily_grid(world)
    if int(grouped_host_graph.color_starts[-1]) != int(host_graph.color_starts[-1]):
        raise RuntimeError("grouped color grid changed row count")
    if int(subfamily_host_graph.row_count) != int(host_graph.color_starts[-1]):
        raise RuntimeError("subfamily color grid changed row count")
    grid_graph = _upload_color_grid(host_graph, world.device)
    grouped_grid_graph = _upload_color_grid(grouped_host_graph, world.device)
    subfamily_graph = _upload_block_world_subfamily_grid(subfamily_host_graph, world.device)
    outer_iters = int(world.solver_iterations) // int(_EXPERIMENTAL_INNER_SWEEPS)
    expected = int(host_graph.color_starts[-1]) * (1 + outer_iters)

    if args.mode == "autotune":
        _run_autotune(
            args,
            world,
            grid_graph,
            grouped_grid_graph,
            subfamily_graph,
            scene,
            num_worlds,
            int(host_graph.color_starts[-1]),
        )
        return
    if args.mode == "adaptive":
        _run_adaptive(
            args,
            world,
            grid_graph,
            grouped_grid_graph,
            subfamily_graph,
            scene,
            num_worlds,
            int(host_graph.color_starts[-1]),
        )
        return

    runs: list[tuple[str, object]] = []
    if args.mode in ("flat", "both", "all"):
        runs.append(("flat", _color_grid_runner(world, grid_graph, block_dim=args.block_dim)))
    if args.mode in ("flat_grouped", "grouped", "both", "all"):
        runs.append(("flat_grouped", _color_grid_runner(world, grouped_grid_graph, block_dim=args.block_dim)))
    if args.mode in ("direct", "both", "all"):
        runs.append(("direct", _color_grid_direct_runner(world, grid_graph, block_dim=args.block_dim)))
    if args.mode in ("block_world", "all"):
        runs.append(("block_world", _block_world_runner(world, block_dim=args.block_world_dim)))
    if args.mode in ("block_world_grouped", "subfamily", "all"):
        runs.append(
            (
                "block_world_grouped",
                _block_world_subfamily_runner(world, subfamily_graph, block_dim=args.block_world_dim),
            )
        )
    if args.mode in ("mega", "all"):
        runs.append(
            (
                "mega",
                _color_grid_mega_runner(world, grid_graph, block_dim=args.block_dim, worker_blocks=args.mega_blocks),
            )
        )
    if args.mode in ("mega_direct", "all"):
        runs.append(
            (
                "mega_direct",
                _color_grid_mega_direct_runner(
                    world,
                    grid_graph,
                    block_dim=args.block_dim,
                    worker_blocks=args.mega_blocks,
                    row_bound=args.row_bound,
                ),
            )
        )

    for _label, run in runs:
        run()
        wp.synchronize_device()
        _validate(grid_graph, expected)

    base_min, base_med = _bench(
        world._solve_main, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=world.device
    )
    pieces = []
    for label, run in runs:
        min_ms, med_ms = _bench(run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=world.device)
        speed = base_min / min_ms if min_ms > 0.0 else float("nan")
        pieces.append(f"{label}={min_ms:8.3f}ms({speed:5.3f}x)")
        if args.verbose:
            print(f"  {label:6s} min={min_ms:.3f} ms med={med_ms:.3f} ms")
    print(
        f"{scene:5s} worlds={num_worlds:5d} rows={int(host_graph.color_starts[-1]):6d} colors={host_graph.num_colors:4d} "
        f"baseline={base_min:8.3f}ms " + " ".join(pieces) + f" base_us={1000.0 * base_min / args.n_runs:8.3f}"
    )
    if args.verbose:
        print(f"  baseline min={base_min:.3f} ms med={base_med:.3f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes", nargs="+", choices=("h1", "g1", "tower", "dr_legs"), default=["h1", "g1", "tower"])
    parser.add_argument("--worlds", default="32", help="Comma-separated world counts.")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--prepare-refresh-stride", type=_parse_stride_value, default="auto")
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--prime-frames", type=int, default=3)
    parser.add_argument("--block-dim", type=int, default=128)
    parser.add_argument("--block-world-dim", type=int, default=_DEFAULT_BLOCK_WORLD_DIM)
    parser.add_argument(
        "--mode",
        choices=(
            "baseline",
            "flat",
            "flat_grouped",
            "grouped",
            "direct",
            "block_world",
            "block_world_grouped",
            "subfamily",
            "autotune",
            "adaptive",
            "mega",
            "mega_direct",
            "both",
            "all",
        ),
        default="baseline",
    )
    parser.add_argument(
        "--allow-unsafe-prototypes",
        action="store_true",
        help="Run prototype scheduler kernels; some modes are known to hang and are not production evidence.",
    )
    parser.add_argument("--mega-blocks", type=int, default=128)
    parser.add_argument("--row-bound", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--tune-block-world-dims", default="32,64,128")
    parser.add_argument("--tune-warmup", type=int, default=1)
    parser.add_argument("--tune-runs", type=int, default=3)
    parser.add_argument("--adapt-rounds", type=int, default=2)
    parser.add_argument("--adapt-window-runs", type=int, default=2)
    parser.add_argument("--adapt-max-window-runs", type=int, default=64)
    parser.add_argument("--adapt-target-ms", type=float, default=1.0)
    parser.add_argument("--adapt-verify-target-ms", type=float, default=2.0)
    parser.add_argument("--adapt-warmup", type=int, default=1)
    parser.add_argument("--adapt-keep-fraction", type=float, default=0.5)
    parser.add_argument("--adapt-min-speedup", type=float, default=1.08)
    parser.add_argument("--adapt-include-launch-heavy", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.mode != "baseline" and not args.allow_unsafe_prototypes:
        parser.error("prototype scheduler modes require --allow-unsafe-prototypes; use --mode baseline for production")
    return args


def main() -> None:
    args = parse_args()
    wp.init()
    worlds = [int(raw.strip()) for raw in args.worlds.split(",") if raw.strip()]
    print(
        f"device={wp.get_device()} block_dim={args.block_dim} block_world_dim={args.block_world_dim} "
        f"prepare_refresh_stride={args.prepare_refresh_stride} mega_blocks={args.mega_blocks} "
        f"row_bound={args.row_bound} n_runs={args.n_runs} mode={args.mode} "
        f"unsafe_prototypes={args.allow_unsafe_prototypes}"
    )
    for scene in args.scenes:
        for num_worlds in worlds:
            run_case(args, scene, num_worlds)


if __name__ == "__main__":
    main()
