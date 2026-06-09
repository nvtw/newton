# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental PhoenX persistent chunk-task solve benchmark.

This benchmark explores a scheduler between the production per-world fast-tail
kernel and expensive per-row readiness. Work is split into chunks of rows from
a single world/color. Persistent worker blocks reserve monotonically increasing
queue slots; when the final chunk of a
color completes, that worker publishes chunks for the next color. Safety is
still color-level, but work distribution is global and can keep more SMs busy
when the number of worlds is small. The queue is intentionally reservation-based
instead of CAS/pop based so idle workers do not repeatedly poll an empty head.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _extract_solver
from newton._src.solvers.phoenx.benchmarks.scenarios import dr_legs, g1_flat, h1_flat, tower
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
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
from newton._src.solvers.phoenx.solver_phoenx_kernels import _FUSED_INNER_SWEEPS, _sync_threads


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__threadfence();
#endif
""")
def _thread_fence(): ...


@dataclass
class ChunkGraphHost:
    chunk_start: np.ndarray
    chunk_count: np.ndarray
    chunk_world: np.ndarray
    chunk_color: np.ndarray
    world_color_chunk_starts: np.ndarray
    world_num_colors: np.ndarray
    max_colors: int
    total_rows: int
    joint_rows: int
    contact_rows: int
    max_color_rows: int
    mean_color_rows: float


@dataclass
class ChunkGraphDevice:
    host: ChunkGraphHost
    chunk_start: wp.array
    chunk_count: wp.array
    chunk_world: wp.array
    chunk_color: wp.array
    world_color_chunk_starts: wp.array
    world_num_colors: wp.array
    remaining_chunks: wp.array
    queue_chunks: wp.array
    queue_epochs: wp.array
    queue_ready: wp.array
    queue_head: wp.array
    queue_tail: wp.array
    total_done: wp.array
    failed: wp.array
    worker_chunk: wp.array
    worker_epoch: wp.array
    queue_capacity: int
    worker_blocks: int


@wp.func
def _chunk_push(
    queue_chunks: wp.array[wp.int32],
    queue_epochs: wp.array[wp.int32],
    queue_ready: wp.array[wp.int32],
    queue_tail: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    queue_capacity: wp.int32,
    chunk: wp.int32,
    epoch: wp.int32,
):
    pos = wp.atomic_add(queue_tail, 0, wp.int32(1))
    if pos < queue_capacity:
        queue_chunks[pos] = epoch * wp.int32(1048576) + chunk
        queue_epochs[pos] = epoch
        _thread_fence()
        queue_ready[pos] = wp.int32(1)
    else:
        failed[0] = wp.int32(2)


@wp.func
def _chunk_reserve(
    queue_chunks: wp.array[wp.int32],
    queue_ready: wp.array[wp.int32],
    queue_head: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    target_total: wp.int32,
    max_wait: wp.int32,
):
    pos = wp.atomic_add(queue_head, 0, wp.int32(1))
    if pos >= target_total:
        return wp.int32(-1)
    wait = wp.int32(0)
    while wp.atomic_add(queue_ready, pos, wp.int32(0)) == wp.int32(0) and wait < max_wait:
        wait = wait + wp.int32(1)
    if wait >= max_wait:
        failed[0] = wp.int32(4)
        return wp.int32(-1)
    task = queue_chunks[pos]
    if task < wp.int32(0):
        failed[0] = wp.int32(3)
        return wp.int32(-1)
    return task


@wp.func
def _chunk_notify_next(
    chunk_world: wp.array[wp.int32],
    chunk_color: wp.array[wp.int32],
    world_color_chunk_starts: wp.array2d[wp.int32],
    world_num_colors: wp.array[wp.int32],
    remaining_chunks: wp.array[wp.int32],
    queue_chunks: wp.array[wp.int32],
    queue_epochs: wp.array[wp.int32],
    queue_ready: wp.array[wp.int32],
    queue_tail: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    chunk: wp.int32,
    epoch: wp.int32,
    num_worlds: wp.int32,
    max_colors: wp.int32,
    num_epochs: wp.int32,
    queue_capacity: wp.int32,
):
    world_id = chunk_world[chunk]
    color = chunk_color[chunk]
    color_index = (epoch * num_worlds + world_id) * max_colors + color
    old = wp.atomic_add(remaining_chunks, color_index, wp.int32(-1))
    if old == wp.int32(1):
        next_color = color + wp.int32(1)
        next_epoch = epoch
        if next_color >= world_num_colors[world_id]:
            next_color = wp.int32(0)
            next_epoch = epoch + wp.int32(1)
        searching = wp.int32(1)
        while next_epoch < num_epochs and searching != wp.int32(0):
            if next_color >= world_num_colors[world_id]:
                next_color = wp.int32(0)
                next_epoch = next_epoch + wp.int32(1)
            else:
                start = world_color_chunk_starts[world_id, next_color]
                end = world_color_chunk_starts[world_id, next_color + wp.int32(1)]
                if start < end:
                    cursor = start
                    while cursor < end:
                        _chunk_push(
                            queue_chunks,
                            queue_epochs,
                            queue_ready,
                            queue_tail,
                            failed,
                            queue_capacity,
                            cursor,
                            next_epoch,
                        )
                        cursor = cursor + wp.int32(1)
                    searching = wp.int32(0)
                else:
                    next_color = next_color + wp.int32(1)
    wp.atomic_add(total_done, 0, wp.int32(1))


@wp.kernel(enable_backward=False)
def _chunk_clear_kernel(
    chunk_count: wp.array[wp.int32],
    world_color_chunk_starts: wp.array2d[wp.int32],
    world_num_colors: wp.array[wp.int32],
    remaining_chunks: wp.array[wp.int32],
    queue_ready: wp.array[wp.int32],
    queue_head: wp.array[wp.int32],
    queue_tail: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    num_worlds: wp.int32,
    max_colors: wp.int32,
    num_epochs: wp.int32,
    queue_capacity: wp.int32,
):
    tid = wp.tid()
    total_colors = num_epochs * num_worlds * max_colors
    if tid < total_colors:
        color = tid - (tid / max_colors) * max_colors
        tmp = tid / max_colors
        world_id = tmp - (tmp / num_worlds) * num_worlds
        if color < world_num_colors[world_id]:
            remaining_chunks[tid] = (
                world_color_chunk_starts[world_id, color + wp.int32(1)] - world_color_chunk_starts[world_id, color]
            )
        else:
            remaining_chunks[tid] = wp.int32(0)
    if tid < queue_capacity:
        queue_ready[tid] = wp.int32(0)
    if tid == wp.int32(0):
        queue_head[0] = wp.int32(0)
        queue_tail[0] = wp.int32(0)
        total_done[0] = wp.int32(0)
        failed[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _chunk_seed_kernel(
    world_color_chunk_starts: wp.array2d[wp.int32],
    world_num_colors: wp.array[wp.int32],
    queue_chunks: wp.array[wp.int32],
    queue_epochs: wp.array[wp.int32],
    queue_ready: wp.array[wp.int32],
    queue_tail: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    num_worlds: wp.int32,
    queue_capacity: wp.int32,
):
    world_id = wp.tid()
    if world_id < num_worlds and world_num_colors[world_id] > wp.int32(0):
        color = wp.int32(0)
        seeded = wp.int32(0)
        while color < world_num_colors[world_id] and seeded == wp.int32(0):
            start = world_color_chunk_starts[world_id, color]
            end = world_color_chunk_starts[world_id, color + wp.int32(1)]
            if start < end:
                cursor = start
                while cursor < end:
                    _chunk_push(
                        queue_chunks,
                        queue_epochs,
                        queue_ready,
                        queue_tail,
                        failed,
                        queue_capacity,
                        cursor,
                        wp.int32(0),
                    )
                    cursor = cursor + wp.int32(1)
                seeded = wp.int32(1)
            color = color + wp.int32(1)


@wp.kernel(enable_backward=False)
def _chunk_prepare_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    chunk_start: wp.array[wp.int32],
    chunk_count: wp.array[wp.int32],
    chunk_world: wp.array[wp.int32],
    chunk_color: wp.array[wp.int32],
    world_color_chunk_starts: wp.array2d[wp.int32],
    world_num_colors: wp.array[wp.int32],
    remaining_chunks: wp.array[wp.int32],
    queue_chunks: wp.array[wp.int32],
    queue_epochs: wp.array[wp.int32],
    queue_ready: wp.array[wp.int32],
    queue_head: wp.array[wp.int32],
    queue_tail: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    worker_chunk: wp.array[wp.int32],
    worker_epoch: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
    num_chunks: wp.int32,
    num_worlds: wp.int32,
    max_colors: wp.int32,
    worker_blocks: wp.int32,
    chunk_threads: wp.int32,
    max_spins: wp.int32,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    queue_capacity: wp.int32,
    copy_state: CopyStateContainer,
):
    tid = wp.tid()
    local_tid = tid - (tid / chunk_threads) * chunk_threads
    worker = tid / chunk_threads
    spin = wp.int32(0)
    done = wp.atomic_add(total_done, 0, wp.int32(0))
    while done < num_chunks and spin < max_spins:
        if local_tid == wp.int32(0):
            task = _chunk_reserve(queue_chunks, queue_ready, queue_head, failed, num_chunks, max_spins)
            claimed = wp.int32(-1)
            if task >= wp.int32(0):
                task_epoch = task / wp.int32(1048576)
                claimed = task - task_epoch * wp.int32(1048576)
                worker_epoch[worker] = task_epoch
            worker_chunk[worker] = claimed
        _sync_threads()
        chunk = worker_chunk[worker]
        epoch = worker_epoch[worker]
        if chunk >= wp.int32(0):
            start = chunk_start[chunk]
            count = chunk_count[chunk]
            cursor = local_tid
            while cursor < count:
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
                cursor = cursor + chunk_threads
            _sync_threads()
            if local_tid == wp.int32(0):
                _chunk_notify_next(
                    chunk_world,
                    chunk_color,
                    world_color_chunk_starts,
                    world_num_colors,
                    remaining_chunks,
                    queue_chunks,
                    queue_epochs,
                    queue_ready,
                    queue_tail,
                    total_done,
                    failed,
                    chunk,
                    epoch,
                    num_worlds,
                    max_colors,
                    wp.int32(1),
                    queue_capacity,
                )
            spin = wp.int32(0)
        else:
            spin = spin + wp.int32(1)
        _sync_threads()
        done = wp.atomic_add(total_done, 0, wp.int32(0))
    if tid == wp.int32(0) and done < num_chunks:
        failed[0] = wp.int32(1)


@wp.kernel(enable_backward=False)
def _chunk_iterate_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    chunk_start: wp.array[wp.int32],
    chunk_count: wp.array[wp.int32],
    chunk_world: wp.array[wp.int32],
    chunk_color: wp.array[wp.int32],
    world_color_chunk_starts: wp.array2d[wp.int32],
    world_num_colors: wp.array[wp.int32],
    remaining_chunks: wp.array[wp.int32],
    queue_chunks: wp.array[wp.int32],
    queue_epochs: wp.array[wp.int32],
    queue_ready: wp.array[wp.int32],
    queue_head: wp.array[wp.int32],
    queue_tail: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    worker_chunk: wp.array[wp.int32],
    worker_epoch: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
    num_chunks: wp.int32,
    num_worlds: wp.int32,
    max_colors: wp.int32,
    num_epochs: wp.int32,
    worker_blocks: wp.int32,
    chunk_threads: wp.int32,
    max_spins: wp.int32,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    inner_sweeps: wp.int32,
    queue_capacity: wp.int32,
    copy_state: CopyStateContainer,
):
    tid = wp.tid()
    local_tid = tid - (tid / chunk_threads) * chunk_threads
    worker = tid / chunk_threads
    target_total = num_chunks * num_epochs
    spin = wp.int32(0)
    done = wp.atomic_add(total_done, 0, wp.int32(0))
    while done < target_total and spin < max_spins:
        if local_tid == wp.int32(0):
            task = _chunk_reserve(queue_chunks, queue_ready, queue_head, failed, target_total, max_spins)
            claimed = wp.int32(-1)
            if task >= wp.int32(0):
                task_epoch = task / wp.int32(1048576)
                claimed = task - task_epoch * wp.int32(1048576)
                worker_epoch[worker] = task_epoch
            worker_chunk[worker] = claimed
        _sync_threads()
        chunk = worker_chunk[worker]
        epoch = worker_epoch[worker]
        if chunk >= wp.int32(0):
            start = chunk_start[chunk]
            count = chunk_count[chunk]
            cursor = local_tid
            while cursor < count:
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
                cursor = cursor + chunk_threads
            _sync_threads()
            if local_tid == wp.int32(0):
                _chunk_notify_next(
                    chunk_world,
                    chunk_color,
                    world_color_chunk_starts,
                    world_num_colors,
                    remaining_chunks,
                    queue_chunks,
                    queue_epochs,
                    queue_ready,
                    queue_tail,
                    total_done,
                    failed,
                    chunk,
                    epoch,
                    num_worlds,
                    max_colors,
                    num_epochs,
                    queue_capacity,
                )
            spin = wp.int32(0)
        else:
            spin = spin + wp.int32(1)
        _sync_threads()
        done = wp.atomic_add(total_done, 0, wp.int32(0))
    if tid == wp.int32(0) and done < target_total:
        failed[0] = wp.int32(1)


@wp.kernel(enable_backward=False)
def _chunk_prepare_serial_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    chunk_start: wp.array[wp.int32],
    chunk_count: wp.array[wp.int32],
    chunk_world: wp.array[wp.int32],
    chunk_color: wp.array[wp.int32],
    world_color_chunk_starts: wp.array2d[wp.int32],
    world_num_colors: wp.array[wp.int32],
    remaining_chunks: wp.array[wp.int32],
    queue_chunks: wp.array[wp.int32],
    queue_epochs: wp.array[wp.int32],
    queue_ready: wp.array[wp.int32],
    queue_head: wp.array[wp.int32],
    queue_tail: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
    num_chunks: wp.int32,
    num_worlds: wp.int32,
    max_colors: wp.int32,
    max_spins: wp.int32,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    queue_capacity: wp.int32,
    copy_state: CopyStateContainer,
):
    spin = wp.int32(0)
    done = wp.atomic_add(total_done, 0, wp.int32(0))
    while done < num_chunks and spin < max_spins:
        claimed = wp.int32(-1)
        epoch = wp.int32(0)
        task = _chunk_reserve(queue_chunks, queue_ready, queue_head, failed, num_chunks, max_spins)
        if task >= wp.int32(0):
            epoch = task / wp.int32(1048576)
            claimed = task - epoch * wp.int32(1048576)
        if claimed >= wp.int32(0):
            start = chunk_start[claimed]
            count = chunk_count[claimed]
            cursor = wp.int32(0)
            while cursor < count:
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
                cursor = cursor + wp.int32(1)
            _chunk_notify_next(
                chunk_world,
                chunk_color,
                world_color_chunk_starts,
                world_num_colors,
                remaining_chunks,
                queue_chunks,
                queue_epochs,
                queue_ready,
                queue_tail,
                total_done,
                failed,
                claimed,
                epoch,
                num_worlds,
                max_colors,
                wp.int32(1),
                queue_capacity,
            )
            spin = wp.int32(0)
        else:
            spin = spin + wp.int32(1)
        done = wp.atomic_add(total_done, 0, wp.int32(0))
    if wp.tid() == wp.int32(0) and done < num_chunks:
        failed[0] = wp.int32(1)


@wp.kernel(enable_backward=False)
def _chunk_iterate_serial_kernel(
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    world_element_ids_by_color: wp.array[wp.int32],
    chunk_start: wp.array[wp.int32],
    chunk_count: wp.array[wp.int32],
    chunk_world: wp.array[wp.int32],
    chunk_color: wp.array[wp.int32],
    world_color_chunk_starts: wp.array2d[wp.int32],
    world_num_colors: wp.array[wp.int32],
    remaining_chunks: wp.array[wp.int32],
    queue_chunks: wp.array[wp.int32],
    queue_epochs: wp.array[wp.int32],
    queue_ready: wp.array[wp.int32],
    queue_head: wp.array[wp.int32],
    queue_tail: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    cc: ContactContainer,
    contacts: ContactViews,
    num_chunks: wp.int32,
    num_worlds: wp.int32,
    max_colors: wp.int32,
    num_epochs: wp.int32,
    max_spins: wp.int32,
    num_joints: wp.int32,
    num_bodies: wp.int32,
    revolute_only: wp.int32,
    inner_sweeps: wp.int32,
    queue_capacity: wp.int32,
    copy_state: CopyStateContainer,
):
    target_total = num_chunks * num_epochs
    spin = wp.int32(0)
    done = wp.atomic_add(total_done, 0, wp.int32(0))
    while done < target_total and spin < max_spins:
        claimed = wp.int32(-1)
        epoch = wp.int32(0)
        task = _chunk_reserve(queue_chunks, queue_ready, queue_head, failed, target_total, max_spins)
        if task >= wp.int32(0):
            epoch = task / wp.int32(1048576)
            claimed = task - epoch * wp.int32(1048576)
        if claimed >= wp.int32(0):
            start = chunk_start[claimed]
            count = chunk_count[claimed]
            cursor = wp.int32(0)
            while cursor < count:
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
                cursor = cursor + wp.int32(1)
            _chunk_notify_next(
                chunk_world,
                chunk_color,
                world_color_chunk_starts,
                world_num_colors,
                remaining_chunks,
                queue_chunks,
                queue_epochs,
                queue_ready,
                queue_tail,
                total_done,
                failed,
                claimed,
                epoch,
                num_worlds,
                max_colors,
                num_epochs,
                queue_capacity,
            )
            spin = wp.int32(0)
        else:
            spin = spin + wp.int32(1)
        done = wp.atomic_add(total_done, 0, wp.int32(0))
    if wp.tid() == wp.int32(0) and done < target_total:
        failed[0] = wp.int32(1)


def _build_scene(scene: str, num_worlds: int, *, substeps: int, solver_iterations: int):
    if scene == "h1":
        return h1_flat.build(num_worlds, "phoenx", substeps, solver_iterations)
    if scene == "g1":
        return g1_flat.build(num_worlds, "phoenx", substeps, solver_iterations)
    if scene == "dr_legs":
        return dr_legs.build(num_worlds, "phoenx", substeps, solver_iterations)
    if scene == "tower":
        return tower.build(num_worlds, "phoenx", substeps, solver_iterations, step_layout="multi_world")
    raise ValueError(f"unknown scene: {scene}")


def _extract_chunk_graph(world: PhoenXWorld, *, chunk_rows: int) -> ChunkGraphHost:
    active = int(world._num_active_constraints.numpy()[0])
    eids_by_color = world._world_element_ids_by_color.numpy()
    starts = world._world_color_starts.numpy()
    csr = world._world_csr_offsets.numpy()
    num_colors = world._world_num_colors.numpy().astype(np.int32, copy=False)
    max_colors = max(1, int(num_colors.max(initial=0)))
    color_starts = np.zeros((world.num_worlds, max_colors + 1), dtype=np.int32)
    chunk_start: list[int] = []
    chunk_count: list[int] = []
    chunk_world: list[int] = []
    chunk_color: list[int] = []
    total_rows = 0
    joint_rows = 0
    contact_rows = 0
    color_row_counts: list[int] = []
    for world_id in range(world.num_worlds):
        base = int(csr[world_id])
        for color in range(max_colors):
            color_starts[world_id, color] = len(chunk_start)
            if color >= int(num_colors[world_id]):
                continue
            start = base + int(starts[world_id, color])
            end = base + int(starts[world_id, color + 1])
            rows = [idx for idx in range(start, end) if 0 <= int(eids_by_color[idx]) < active]
            total_rows += len(rows)
            color_row_counts.append(len(rows))
            for row in rows:
                if int(eids_by_color[row]) < int(world.num_joints):
                    joint_rows += 1
                else:
                    contact_rows += 1
            for offset in range(0, len(rows), chunk_rows):
                # The source rows are contiguous after filtering in the rigid scenes this
                # benchmark targets. Keep the assertion here so the chunk kernel can use
                # a start/count pair instead of a second indirection table.
                chunk = rows[offset : offset + chunk_rows]
                if not chunk:
                    continue
                if chunk[-1] - chunk[0] + 1 != len(chunk):
                    raise RuntimeError("non-contiguous filtered chunk; add a chunk-index indirection")
                chunk_start.append(int(chunk[0]))
                chunk_count.append(len(chunk))
                chunk_world.append(world_id)
                chunk_color.append(color)
            color_starts[world_id, color + 1] = len(chunk_start)
        for color in range(int(num_colors[world_id]), max_colors):
            color_starts[world_id, color + 1] = len(chunk_start)
    return ChunkGraphHost(
        chunk_start=np.asarray(chunk_start, dtype=np.int32),
        chunk_count=np.asarray(chunk_count, dtype=np.int32),
        chunk_world=np.asarray(chunk_world, dtype=np.int32),
        chunk_color=np.asarray(chunk_color, dtype=np.int32),
        world_color_chunk_starts=color_starts,
        world_num_colors=np.asarray(num_colors, dtype=np.int32),
        max_colors=max_colors,
        total_rows=total_rows,
        joint_rows=joint_rows,
        contact_rows=contact_rows,
        max_color_rows=max(color_row_counts, default=0),
        mean_color_rows=float(np.mean(np.asarray(color_row_counts, dtype=np.float32))) if color_row_counts else 0.0,
    )


def _upload_chunk_graph(
    graph: ChunkGraphHost,
    device: wp.context.Devicelike,
    *,
    max_epochs: int,
    worker_blocks: int,
) -> ChunkGraphDevice:
    num_chunks = max(1, int(graph.chunk_start.shape[0]))
    queue_capacity = max(1, num_chunks * max(1, int(max_epochs)))
    return ChunkGraphDevice(
        host=graph,
        chunk_start=wp.array(graph.chunk_start, dtype=wp.int32, device=device),
        chunk_count=wp.array(graph.chunk_count, dtype=wp.int32, device=device),
        chunk_world=wp.array(graph.chunk_world, dtype=wp.int32, device=device),
        chunk_color=wp.array(graph.chunk_color, dtype=wp.int32, device=device),
        world_color_chunk_starts=wp.array(graph.world_color_chunk_starts, dtype=wp.int32, device=device),
        world_num_colors=wp.array(graph.world_num_colors, dtype=wp.int32, device=device),
        remaining_chunks=wp.zeros(
            max(1, max_epochs * graph.world_num_colors.shape[0] * graph.max_colors), dtype=wp.int32, device=device
        ),
        queue_chunks=wp.zeros(queue_capacity, dtype=wp.int32, device=device),
        queue_epochs=wp.zeros(queue_capacity, dtype=wp.int32, device=device),
        queue_ready=wp.zeros(queue_capacity, dtype=wp.int32, device=device),
        queue_head=wp.zeros(1, dtype=wp.int32, device=device),
        queue_tail=wp.zeros(1, dtype=wp.int32, device=device),
        total_done=wp.zeros(1, dtype=wp.int32, device=device),
        failed=wp.zeros(1, dtype=wp.int32, device=device),
        worker_chunk=wp.zeros(max(1, worker_blocks), dtype=wp.int32, device=device),
        worker_epoch=wp.zeros(max(1, worker_blocks), dtype=wp.int32, device=device),
        queue_capacity=queue_capacity,
        worker_blocks=worker_blocks,
    )


def _reset_chunk_graph(graph: ChunkGraphDevice, device: wp.context.Devicelike, *, num_epochs: int) -> None:
    wp.launch(
        _chunk_clear_kernel,
        dim=max(
            graph.queue_capacity, int(num_epochs) * graph.host.world_num_colors.shape[0] * graph.host.max_colors, 1
        ),
        inputs=[
            graph.chunk_count,
            graph.world_color_chunk_starts,
            graph.world_num_colors,
            graph.remaining_chunks,
            graph.queue_ready,
            graph.queue_head,
            graph.queue_tail,
            graph.total_done,
            graph.failed,
            wp.int32(graph.host.world_num_colors.shape[0]),
            wp.int32(graph.host.max_colors),
            wp.int32(num_epochs),
            wp.int32(graph.queue_capacity),
        ],
        device=device,
    )
    wp.launch(
        _chunk_seed_kernel,
        dim=max(1, graph.host.world_num_colors.shape[0]),
        inputs=[
            graph.world_color_chunk_starts,
            graph.world_num_colors,
            graph.queue_chunks,
            graph.queue_epochs,
            graph.queue_ready,
            graph.queue_tail,
            graph.failed,
            wp.int32(graph.host.world_num_colors.shape[0]),
            wp.int32(graph.queue_capacity),
        ],
        device=device,
    )


def _chunk_runner(world: PhoenXWorld, graph: ChunkGraphDevice, *, chunk_threads: int, max_spins: int):
    device = world.device
    contact_views = world._contact_views if world._contact_views is not None else world._contact_views_placeholder
    idt = wp.float32(1.0 / world.substep_dt)
    inner_sweeps = int(_FUSED_INNER_SWEEPS)
    outer_iters = int(world.solver_iterations) // inner_sweeps
    revolute_only = 1 if bool(world._use_revolute_specialization) else 0
    dim = max(1, graph.worker_blocks * int(chunk_threads))

    def run() -> None:
        _reset_chunk_graph(graph, device, num_epochs=1)
        if int(chunk_threads) == 1:
            wp.launch(
                _chunk_prepare_serial_kernel,
                dim=max(1, graph.worker_blocks),
                inputs=[
                    world.constraints,
                    world._contact_cols,
                    world.bodies,
                    world._particles_or_sentinel(),
                    idt,
                    world._world_element_ids_by_color,
                    graph.chunk_start,
                    graph.chunk_count,
                    graph.chunk_world,
                    graph.chunk_color,
                    graph.world_color_chunk_starts,
                    graph.world_num_colors,
                    graph.remaining_chunks,
                    graph.queue_chunks,
                    graph.queue_epochs,
                    graph.queue_ready,
                    graph.queue_head,
                    graph.queue_tail,
                    graph.total_done,
                    graph.failed,
                    world._contact_container,
                    contact_views,
                    wp.int32(graph.host.chunk_start.shape[0]),
                    wp.int32(world.num_worlds),
                    wp.int32(graph.host.max_colors),
                    wp.int32(max_spins),
                    wp.int32(world.num_joints),
                    wp.int32(world.num_bodies),
                    wp.int32(revolute_only),
                    wp.int32(graph.queue_capacity),
                    world._copy_state,
                ],
                device=device,
            )
        else:
            wp.launch(
                _chunk_prepare_kernel,
                dim=dim,
                block_dim=chunk_threads,
                inputs=[
                    world.constraints,
                    world._contact_cols,
                    world.bodies,
                    world._particles_or_sentinel(),
                    idt,
                    world._world_element_ids_by_color,
                    graph.chunk_start,
                    graph.chunk_count,
                    graph.chunk_world,
                    graph.chunk_color,
                    graph.world_color_chunk_starts,
                    graph.world_num_colors,
                    graph.remaining_chunks,
                    graph.queue_chunks,
                    graph.queue_epochs,
                    graph.queue_ready,
                    graph.queue_head,
                    graph.queue_tail,
                    graph.total_done,
                    graph.failed,
                    graph.worker_chunk,
                    graph.worker_epoch,
                    world._contact_container,
                    contact_views,
                    wp.int32(graph.host.chunk_start.shape[0]),
                    wp.int32(world.num_worlds),
                    wp.int32(graph.host.max_colors),
                    wp.int32(graph.worker_blocks),
                    wp.int32(chunk_threads),
                    wp.int32(max_spins),
                    wp.int32(world.num_joints),
                    wp.int32(world.num_bodies),
                    wp.int32(revolute_only),
                    wp.int32(graph.queue_capacity),
                    world._copy_state,
                ],
                device=device,
            )
        _reset_chunk_graph(graph, device, num_epochs=outer_iters)
        if int(chunk_threads) == 1:
            wp.launch(
                _chunk_iterate_serial_kernel,
                dim=max(1, graph.worker_blocks),
                inputs=[
                    world.constraints,
                    world._contact_cols,
                    world.bodies,
                    world._particles_or_sentinel(),
                    idt,
                    wp.float32(world.sor_boost),
                    world._world_element_ids_by_color,
                    graph.chunk_start,
                    graph.chunk_count,
                    graph.chunk_world,
                    graph.chunk_color,
                    graph.world_color_chunk_starts,
                    graph.world_num_colors,
                    graph.remaining_chunks,
                    graph.queue_chunks,
                    graph.queue_epochs,
                    graph.queue_ready,
                    graph.queue_head,
                    graph.queue_tail,
                    graph.total_done,
                    graph.failed,
                    world._contact_container,
                    contact_views,
                    wp.int32(graph.host.chunk_start.shape[0]),
                    wp.int32(world.num_worlds),
                    wp.int32(graph.host.max_colors),
                    wp.int32(outer_iters),
                    wp.int32(max_spins),
                    wp.int32(world.num_joints),
                    wp.int32(world.num_bodies),
                    wp.int32(revolute_only),
                    wp.int32(inner_sweeps),
                    wp.int32(graph.queue_capacity),
                    world._copy_state,
                ],
                device=device,
            )
        else:
            wp.launch(
                _chunk_iterate_kernel,
                dim=dim,
                block_dim=chunk_threads,
                inputs=[
                    world.constraints,
                    world._contact_cols,
                    world.bodies,
                    world._particles_or_sentinel(),
                    idt,
                    wp.float32(world.sor_boost),
                    world._world_element_ids_by_color,
                    graph.chunk_start,
                    graph.chunk_count,
                    graph.chunk_world,
                    graph.chunk_color,
                    graph.world_color_chunk_starts,
                    graph.world_num_colors,
                    graph.remaining_chunks,
                    graph.queue_chunks,
                    graph.queue_epochs,
                    graph.queue_ready,
                    graph.queue_head,
                    graph.queue_tail,
                    graph.total_done,
                    graph.failed,
                    graph.worker_chunk,
                    graph.worker_epoch,
                    world._contact_container,
                    contact_views,
                    wp.int32(graph.host.chunk_start.shape[0]),
                    wp.int32(world.num_worlds),
                    wp.int32(graph.host.max_colors),
                    wp.int32(outer_iters),
                    wp.int32(graph.worker_blocks),
                    wp.int32(chunk_threads),
                    wp.int32(max_spins),
                    wp.int32(world.num_joints),
                    wp.int32(world.num_bodies),
                    wp.int32(revolute_only),
                    wp.int32(inner_sweeps),
                    wp.int32(graph.queue_capacity),
                    world._copy_state,
                ],
                device=device,
            )

    return run


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


def _validate(graph: ChunkGraphDevice, expected: int, label: str) -> None:
    total = int(graph.total_done.numpy()[0])
    failed = int(graph.failed.numpy()[0])
    if total != expected or failed != 0:
        head = int(graph.queue_head.numpy()[0])
        tail = int(graph.queue_tail.numpy()[0])
        remaining = graph.remaining_chunks.numpy()
        nonzero = np.flatnonzero(remaining)
        positive = np.flatnonzero(remaining > 0)
        negative = np.flatnonzero(remaining < 0)
        first = int(nonzero[0]) if nonzero.size else -1
        first_value = int(remaining[first]) if first >= 0 else 0
        first_positive = int(positive[0]) if positive.size else -1
        first_positive_value = int(remaining[first_positive]) if first_positive >= 0 else 0
        first_negative = int(negative[0]) if negative.size else -1
        first_negative_value = int(remaining[first_negative]) if first_negative >= 0 else 0
        raise RuntimeError(
            f"{label} failed: total_done={total} failed={failed} expected={expected} "
            f"queue_head={head} queue_tail={tail} remaining_nonzero={nonzero.size} "
            f"first_remaining={first}:{first_value} first_positive={first_positive}:{first_positive_value} "
            f"first_negative={first_negative}:{first_negative_value}"
        )


def _select_chunk_chain(args: argparse.Namespace, graph: ChunkGraphHost, num_worlds: int) -> bool:
    rows_per_world = float(graph.total_rows) / float(max(1, num_worlds))
    contact_fraction = float(graph.contact_rows) / float(max(1, graph.total_rows))
    chunks_per_world_color = float(graph.chunk_start.shape[0]) / float(max(1, num_worlds * graph.max_colors))
    return (
        rows_per_world >= args.hybrid_min_rows_per_world
        and contact_fraction >= args.hybrid_min_contact_fraction
        and chunks_per_world_color >= args.hybrid_min_chunks_per_world_color
    )


def run_case(args: argparse.Namespace, scene: str, num_worlds: int) -> None:
    handle = _build_scene(scene, num_worlds, substeps=args.substeps, solver_iterations=args.solver_iterations)
    solver = _extract_solver(handle)
    world = solver.world
    if world.step_layout == "single_world" or world.mass_splitting_enabled or world.num_particles > 0:
        raise RuntimeError("chunk-task prototype only supports multi_world rigid scenes")
    if world._has_soft_contact_pd or world._sleeping_enabled:
        raise RuntimeError("chunk-task prototype currently excludes soft-contact PD and sleeping")
    for _ in range(args.prime_frames):
        handle.simulate_one_frame()
    wp.synchronize_device()

    outer_iters = int(world.solver_iterations) // int(_FUSED_INNER_SWEEPS)
    host_graph = _extract_chunk_graph(world, chunk_rows=args.chunk_rows)
    chunk_graph = _upload_chunk_graph(
        host_graph, world.device, max_epochs=max(1, outer_iters), worker_blocks=args.worker_blocks
    )
    chunk_run = _chunk_runner(world, chunk_graph, chunk_threads=args.chunk_threads, max_spins=args.max_spins)

    chunk_run()
    wp.synchronize_device()
    _validate(chunk_graph, int(host_graph.chunk_start.shape[0]) * outer_iters, "chunk iterate")

    base_min, base_med = _bench(
        world._solve_main, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=world.device
    )
    chunk_min, chunk_med = _bench(
        chunk_run, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials, device=world.device
    )
    speed = base_min / chunk_min if chunk_min > 0.0 else float("nan")
    rows_per_world = float(host_graph.total_rows) / float(max(1, num_worlds))
    contact_fraction = float(host_graph.contact_rows) / float(max(1, host_graph.total_rows))
    chunks_per_world_color = float(host_graph.chunk_start.shape[0]) / float(max(1, num_worlds * host_graph.max_colors))
    use_chain = _select_chunk_chain(args, host_graph, num_worlds)
    hybrid_min = chunk_min if use_chain else base_min
    hybrid_speed = base_min / hybrid_min if hybrid_min > 0.0 else float("nan")
    hybrid_choice = "chain" if use_chain else "base"
    print(
        f"{scene:7s} worlds={num_worlds:5d} rows={host_graph.total_rows:6d} chunks={host_graph.chunk_start.shape[0]:6d} "
        f"colors={host_graph.max_colors:3d} rowpw={rows_per_world:7.1f} contact_frac={contact_fraction:5.2f} "
        f"chunkpc={chunks_per_world_color:5.2f} baseline={base_min:8.3f}ms chain={chunk_min:8.3f}ms "
        f"speedup={speed:6.3f}x hybrid={hybrid_choice:5s} hybrid_speedup={hybrid_speed:6.3f}x "
        f"base_us={1000.0 * base_min / args.n_runs:8.3f} chain_us={1000.0 * chunk_min / args.n_runs:8.3f}"
    )
    if args.verbose:
        print(f"  med baseline={base_med:.3f} ms chain={chunk_med:.3f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scenes", nargs="+", choices=("h1", "g1", "dr_legs", "tower"), default=["h1", "g1", "dr_legs", "tower"]
    )
    parser.add_argument("--worlds", default="32", help="Comma-separated world counts.")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--prime-frames", type=int, default=3)
    parser.add_argument("--chunk-rows", type=int, default=32)
    parser.add_argument("--chunk-threads", type=int, default=32)
    parser.add_argument("--worker-blocks", type=int, default=64)
    parser.add_argument("--max-spins", type=int, default=1048576)
    parser.add_argument("--hybrid-min-rows-per-world", type=float, default=256.0)
    parser.add_argument("--hybrid-min-contact-fraction", type=float, default=0.5)
    parser.add_argument("--hybrid-min-chunks-per-world-color", type=float, default=1.5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    worlds = [int(raw.strip()) for raw in args.worlds.split(",") if raw.strip()]
    print(
        f"device={wp.get_device()} chunk_rows={args.chunk_rows} chunk_threads={args.chunk_threads} "
        f"worker_blocks={args.worker_blocks} n_runs={args.n_runs}"
    )
    for scene in args.scenes:
        for num_worlds in worlds:
            run_case(args, scene, num_worlds)


if __name__ == "__main__":
    main()
