# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Synthetic PhoenX megakernel scheduling benchmark.

This is intentionally not a solver dispatch path. It compares the scheduling
shape from ``SchedulingExperiments.Cuda`` against a conventional colored PGS
loop on synthetic interaction graphs:

* ``colored`` runs one captured kernel node per color and epoch. This is a
  naive launch-bound baseline, not PhoenX's production fast-tail path;
* ``node_ready`` runs one persistent scan kernel that uses per-node epoch
  counters and atomics to decide when each interaction may execute;
* ``world_ready`` uses the same dependency rule, but each fixed-size lane
  group scans only one world's rows, matching PhoenX's multi-world fast-tail
  layout more closely.

Each work item now carries a PhoenX-like descriptor: family tag, endpoint
count, row/block width, and projection mode. The payload is still arithmetic
(not real constraint math), but it can run either a divergent specialized path
or a fixed-shape unified path. This keeps the benchmark focused on whether
megakernel scheduling and unified descriptor fetch are plausible before we wire
them into real joint/contact rows.

The ``--real-scenes`` option extracts the active colored interaction graph from
small PhoenX benchmark scenes and converts it into the same descriptor stream.

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.experimental.bench_megakernel_scheduling
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _bench, _extract_solver
from newton._src.solvers.phoenx.benchmarks.scenarios import g1_flat, h1_flat, tower

MAX_NODES_PER_INTERACTION = 8
PAYLOAD_SPECIALIZED = 0
PAYLOAD_UNIFIED = 1


@dataclass(frozen=True)
class SyntheticGraph:
    name: str
    nodes: np.ndarray
    node_count: np.ndarray
    node_start: np.ndarray
    node_stride: np.ndarray
    world_starts: np.ndarray
    world_color_starts: np.ndarray
    family: np.ndarray
    row_size: np.ndarray
    projection: np.ndarray
    color_buffer: np.ndarray
    color_starts: np.ndarray
    num_nodes: int


@wp.func
def _simulate_payload(work_iters: wp.int32, seed: wp.int32) -> wp.float32:
    x = wp.float32(1.0) + wp.float32(seed & wp.int32(1023)) * wp.float32(0.001)
    y = wp.float32(2.0) + wp.float32((seed >> wp.int32(10)) & wp.int32(1023)) * wp.float32(0.001)
    k = wp.int32(0)
    while k < work_iters:
        x = x * wp.float32(1.000001) + y * wp.float32(0.000001)
        y = y * wp.float32(0.999999) + x * wp.float32(0.000001)
        k = k + wp.int32(1)
    return x + y


@wp.func
def _project_payload(value: wp.float32, projection: wp.int32) -> wp.float32:
    if projection == wp.int32(0):
        return value
    if projection == wp.int32(1):
        return wp.min(wp.max(value, wp.float32(-4.0)), wp.float32(4.0))
    limit = wp.float32(2.0) + wp.float32(projection) * wp.float32(0.25)
    denom = wp.abs(value) + wp.float32(1.0e-6)
    return value * wp.min(wp.float32(1.0), limit / denom)


@wp.func
def _descriptor_payload(
    work_iters: wp.int32,
    seed: wp.int32,
    family: wp.int32,
    row_size: wp.int32,
    projection: wp.int32,
    max_rows: wp.int32,
    payload_mode: wp.int32,
) -> wp.float32:
    value = wp.float32(0.0)
    if payload_mode == wp.int32(PAYLOAD_UNIFIED):
        row = wp.int32(0)
        while row < max_rows:
            active = wp.float32(0.0)
            if row < row_size:
                active = wp.float32(1.0)
            value += active * _project_payload(_simulate_payload(work_iters, seed + row * wp.int32(131)), projection)
            row = row + wp.int32(1)
        return value

    if family == wp.int32(0):
        row = wp.int32(0)
        while row < row_size:
            value += _simulate_payload(work_iters, seed + row * wp.int32(131))
            row = row + wp.int32(1)
        return value
    if family == wp.int32(1):
        row = wp.int32(0)
        while row < row_size:
            value += _project_payload(_simulate_payload(work_iters, seed + row * wp.int32(131)), projection)
            row = row + wp.int32(1)
        return value

    row = wp.int32(0)
    while row < row_size:
        value += _project_payload(
            _simulate_payload(work_iters, seed + row * wp.int32(131)),
            projection + family,
        )
        row = row + wp.int32(1)
    return value


@wp.func
def _nodes_ready(
    nodes: wp.array2d[wp.int32],
    node_count: wp.array[wp.int32],
    node_start: wp.array2d[wp.int32],
    node_stride: wp.array2d[wp.int32],
    per_node_done: wp.array[wp.int32],
    interaction_index: wp.int32,
    epoch: wp.int32,
) -> bool:
    ready = wp.int32(1)
    n = node_count[interaction_index]
    slot = wp.int32(0)
    while slot < n:
        node_id = nodes[slot, interaction_index]
        required = node_start[slot, interaction_index] + epoch * node_stride[slot, interaction_index]
        observed = wp.atomic_add(per_node_done, node_id, wp.int32(0))
        if observed != required:
            ready = wp.int32(0)
        slot = slot + wp.int32(1)
    return ready != wp.int32(0)


@wp.kernel(enable_backward=False)
def _reset_scheduler_kernel(
    per_node_done: wp.array[wp.int32],
    next_executable: wp.array[wp.int32],
    world_done: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    sink: wp.array[wp.float32],
    num_nodes: wp.int32,
    num_interactions: wp.int32,
    num_worlds: wp.int32,
):
    tid = wp.tid()
    if tid < num_nodes:
        per_node_done[tid] = wp.int32(0)
    if tid < num_interactions:
        next_executable[tid] = wp.int32(0)
    if tid < num_worlds:
        world_done[tid] = wp.int32(0)
    if tid == wp.int32(0):
        total_done[0] = wp.int32(0)
        failed[0] = wp.int32(0)
        sink[0] = wp.float32(0.0)


@wp.kernel(enable_backward=False)
def _colored_color_kernel(
    nodes: wp.array2d[wp.int32],
    node_count: wp.array[wp.int32],
    family: wp.array[wp.int32],
    row_size: wp.array[wp.int32],
    projection: wp.array[wp.int32],
    color_buffer: wp.array[wp.int32],
    per_node_done: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    sink: wp.array[wp.float32],
    color_start: wp.int32,
    work_iters: wp.int32,
    max_rows: wp.int32,
    payload_mode: wp.int32,
):
    lane = wp.tid()
    interaction_index = color_buffer[color_start + lane]
    node_count_i = node_count[interaction_index]
    seed = interaction_index + total_done[0] * wp.int32(17)
    value = _descriptor_payload(
        work_iters,
        seed,
        family[interaction_index],
        row_size[interaction_index],
        projection[interaction_index],
        max_rows,
        payload_mode,
    )
    wp.atomic_add(sink, 0, value)

    slot = wp.int32(0)
    while slot < node_count_i:
        node_id = nodes[slot, interaction_index]
        per_node_done[node_id] = per_node_done[node_id] + wp.int32(1)
        slot = slot + wp.int32(1)
    wp.atomic_add(total_done, 0, wp.int32(1))


@wp.kernel(enable_backward=False)
def _overlap_color_kernel(
    nodes: wp.array2d[wp.int32],
    node_count: wp.array[wp.int32],
    node_start: wp.array2d[wp.int32],
    node_stride: wp.array2d[wp.int32],
    family: wp.array[wp.int32],
    row_size: wp.array[wp.int32],
    projection: wp.array[wp.int32],
    color_buffer: wp.array[wp.int32],
    per_node_done: wp.array[wp.int32],
    next_executable: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    sink: wp.array[wp.float32],
    color_start: wp.int32,
    epoch: wp.int32,
    work_iters: wp.int32,
    max_spin_passes: wp.int32,
    max_rows: wp.int32,
    payload_mode: wp.int32,
):
    lane = wp.tid()
    interaction_index = color_buffer[color_start + lane]
    spin = wp.int32(0)
    executed = wp.int32(0)
    while executed == wp.int32(0) and spin < max_spin_passes:
        row_epoch = wp.atomic_add(next_executable, interaction_index, wp.int32(0))
        if row_epoch > epoch:
            executed = wp.int32(1)
        elif row_epoch == epoch:
            if _nodes_ready(nodes, node_count, node_start, node_stride, per_node_done, interaction_index, epoch):
                if _nodes_ready(nodes, node_count, node_start, node_stride, per_node_done, interaction_index, epoch):
                    old = wp.atomic_cas(next_executable, interaction_index, epoch, epoch + wp.int32(1))
                    if old == epoch:
                        seed = interaction_index + epoch * wp.int32(65537)
                        value = _descriptor_payload(
                            work_iters,
                            seed,
                            family[interaction_index],
                            row_size[interaction_index],
                            projection[interaction_index],
                            max_rows,
                            payload_mode,
                        )
                        wp.atomic_add(sink, 0, value)

                        n = node_count[interaction_index]
                        slot = wp.int32(0)
                        while slot < n:
                            node_id = nodes[slot, interaction_index]
                            wp.atomic_add(per_node_done, node_id, wp.int32(1))
                            slot = slot + wp.int32(1)
                        wp.atomic_add(total_done, 0, wp.int32(1))
                        executed = wp.int32(1)
        spin = spin + wp.int32(1)

    if executed == wp.int32(0):
        failed[0] = wp.int32(1)


@wp.kernel(enable_backward=False)
def _node_ready_scan_kernel(
    nodes: wp.array2d[wp.int32],
    node_count: wp.array[wp.int32],
    node_start: wp.array2d[wp.int32],
    node_stride: wp.array2d[wp.int32],
    family: wp.array[wp.int32],
    row_size: wp.array[wp.int32],
    projection: wp.array[wp.int32],
    per_node_done: wp.array[wp.int32],
    next_executable: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    sink: wp.array[wp.float32],
    num_interactions: wp.int32,
    num_iterations: wp.int32,
    total_threads: wp.int32,
    work_iters: wp.int32,
    max_passes: wp.int32,
    max_rows: wp.int32,
    payload_mode: wp.int32,
):
    tid = wp.tid()
    target_total = num_interactions * num_iterations
    pass_count = wp.int32(0)
    done = wp.atomic_add(total_done, 0, wp.int32(0))
    while done < target_total and pass_count < max_passes:
        interaction_index = tid
        while interaction_index < num_interactions:
            epoch = next_executable[interaction_index]
            if epoch < num_iterations:
                if _nodes_ready(nodes, node_count, node_start, node_stride, per_node_done, interaction_index, epoch):
                    if _nodes_ready(
                        nodes, node_count, node_start, node_stride, per_node_done, interaction_index, epoch
                    ):
                        old = wp.atomic_cas(next_executable, interaction_index, epoch, epoch + wp.int32(1))
                        if old == epoch:
                            seed = interaction_index + epoch * wp.int32(65537)
                            value = _descriptor_payload(
                                work_iters,
                                seed,
                                family[interaction_index],
                                row_size[interaction_index],
                                projection[interaction_index],
                                max_rows,
                                payload_mode,
                            )
                            wp.atomic_add(sink, 0, value)

                            n = node_count[interaction_index]
                            slot = wp.int32(0)
                            while slot < n:
                                node_id = nodes[slot, interaction_index]
                                wp.atomic_add(per_node_done, node_id, wp.int32(1))
                                slot = slot + wp.int32(1)
                            wp.atomic_add(total_done, 0, wp.int32(1))
            interaction_index = interaction_index + total_threads
        pass_count = pass_count + wp.int32(1)
        done = wp.atomic_add(total_done, 0, wp.int32(0))

    if tid == wp.int32(0) and done < target_total:
        failed[0] = wp.int32(1)


@wp.kernel(enable_backward=False)
def _world_ready_scan_kernel(
    nodes: wp.array2d[wp.int32],
    node_count: wp.array[wp.int32],
    node_start: wp.array2d[wp.int32],
    node_stride: wp.array2d[wp.int32],
    family: wp.array[wp.int32],
    row_size: wp.array[wp.int32],
    projection: wp.array[wp.int32],
    world_starts: wp.array[wp.int32],
    per_node_done: wp.array[wp.int32],
    next_executable: wp.array[wp.int32],
    world_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    sink: wp.array[wp.float32],
    num_worlds: wp.int32,
    num_iterations: wp.int32,
    threads_per_world: wp.int32,
    work_iters: wp.int32,
    max_passes: wp.int32,
    max_rows: wp.int32,
    payload_mode: wp.int32,
):
    tid = wp.tid()
    local_tid = tid % threads_per_world
    world_id = tid / threads_per_world
    if world_id >= num_worlds:
        return

    begin = world_starts[world_id]
    end = world_starts[world_id + wp.int32(1)]
    target_total = (end - begin) * num_iterations
    pass_count = wp.int32(0)
    done = wp.atomic_add(world_done, world_id, wp.int32(0))
    while done < target_total and pass_count < max_passes:
        interaction_index = begin + local_tid
        while interaction_index < end:
            epoch = next_executable[interaction_index]
            if epoch < num_iterations:
                if _nodes_ready(nodes, node_count, node_start, node_stride, per_node_done, interaction_index, epoch):
                    if _nodes_ready(
                        nodes, node_count, node_start, node_stride, per_node_done, interaction_index, epoch
                    ):
                        old = wp.atomic_cas(next_executable, interaction_index, epoch, epoch + wp.int32(1))
                        if old == epoch:
                            seed = interaction_index + epoch * wp.int32(65537)
                            value = _descriptor_payload(
                                work_iters,
                                seed,
                                family[interaction_index],
                                row_size[interaction_index],
                                projection[interaction_index],
                                max_rows,
                                payload_mode,
                            )
                            wp.atomic_add(sink, 0, value)

                            n = node_count[interaction_index]
                            slot = wp.int32(0)
                            while slot < n:
                                node_id = nodes[slot, interaction_index]
                                wp.atomic_add(per_node_done, node_id, wp.int32(1))
                                slot = slot + wp.int32(1)
                            wp.atomic_add(world_done, world_id, wp.int32(1))
            interaction_index = interaction_index + threads_per_world
        pass_count = pass_count + wp.int32(1)
        done = wp.atomic_add(world_done, world_id, wp.int32(0))

    if local_tid == wp.int32(0) and done < target_total:
        failed[0] = wp.int32(1)


@wp.kernel(enable_backward=False)
def _reset_window_scheduler_kernel(
    per_node_done: wp.array[wp.int32],
    next_executable: wp.array[wp.int32],
    world_done: wp.array[wp.int32],
    window_phase: wp.array[wp.int32],
    window_color: wp.array[wp.int32],
    window_color_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    sink: wp.array[wp.float32],
    num_nodes: wp.int32,
    num_interactions: wp.int32,
    num_worlds: wp.int32,
    total_color_epochs: wp.int32,
):
    tid = wp.tid()
    if tid < num_nodes:
        per_node_done[tid] = wp.int32(0)
    if tid < num_interactions:
        next_executable[tid] = wp.int32(0)
    if tid < num_worlds:
        world_done[tid] = wp.int32(0)
        window_phase[tid] = wp.int32(0)
        window_color[tid] = wp.int32(0)
    if tid < total_color_epochs:
        window_color_done[tid] = wp.int32(0)
    if tid == wp.int32(0):
        failed[0] = wp.int32(0)
        sink[0] = wp.float32(0.0)


@wp.kernel(enable_backward=False)
def _world_window_ready_kernel(
    nodes: wp.array2d[wp.int32],
    node_count: wp.array[wp.int32],
    node_start: wp.array2d[wp.int32],
    node_stride: wp.array2d[wp.int32],
    family: wp.array[wp.int32],
    row_size: wp.array[wp.int32],
    projection: wp.array[wp.int32],
    color_buffer: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    world_color_starts: wp.array[wp.int32],
    per_node_done: wp.array[wp.int32],
    next_executable: wp.array[wp.int32],
    world_done: wp.array[wp.int32],
    window_phase: wp.array[wp.int32],
    window_color: wp.array[wp.int32],
    window_color_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    sink: wp.array[wp.float32],
    num_worlds: wp.int32,
    num_colors: wp.int32,
    num_iterations: wp.int32,
    threads_per_world: wp.int32,
    window_colors: wp.int32,
    work_iters: wp.int32,
    max_passes: wp.int32,
    max_rows: wp.int32,
    payload_mode: wp.int32,
):
    tid = wp.tid()
    local_tid = tid % threads_per_world
    world_id = tid / threads_per_world
    if world_id >= num_worlds:
        return

    color_begin = world_color_starts[world_id]
    color_end = world_color_starts[world_id + wp.int32(1)]
    local_color_count = color_end - color_begin
    if local_color_count <= wp.int32(0):
        return

    target_total = (color_starts[color_end] - color_starts[color_begin]) * num_iterations
    pass_count = wp.int32(0)
    done = wp.atomic_add(world_done, world_id, wp.int32(0))
    while done < target_total and pass_count < max_passes:
        if local_tid == wp.int32(0):
            head_epoch = wp.atomic_add(window_phase, world_id, wp.int32(0))
            head_color = wp.atomic_add(window_color, world_id, wp.int32(0))
            keep_advancing = wp.int32(1)
            while keep_advancing != wp.int32(0) and head_epoch < num_iterations:
                if head_color >= local_color_count:
                    head_epoch = head_epoch + wp.int32(1)
                    head_color = wp.int32(0)
                else:
                    global_color = color_begin + head_color
                    color_count = color_starts[global_color + wp.int32(1)] - color_starts[global_color]
                    color_epoch = head_epoch * num_colors + global_color
                    observed = wp.atomic_add(window_color_done, color_epoch, wp.int32(0))
                    if observed >= color_count:
                        head_color = head_color + wp.int32(1)
                    else:
                        keep_advancing = wp.int32(0)
            window_phase[world_id] = head_epoch
            window_color[world_id] = head_color

        head_epoch = wp.atomic_add(window_phase, world_id, wp.int32(0))
        head_color = wp.atomic_add(window_color, world_id, wp.int32(0))
        window_offset = wp.int32(0)
        while window_offset < window_colors:
            local_color = head_color + window_offset
            epoch = head_epoch
            if local_color >= local_color_count:
                local_color = local_color - local_color_count
                epoch = epoch + wp.int32(1)
            if epoch < num_iterations:
                global_color = color_begin + local_color
                start = color_starts[global_color]
                end = color_starts[global_color + wp.int32(1)]
                color_epoch = epoch * num_colors + global_color
                cursor = start + local_tid
                while cursor < end:
                    interaction_index = color_buffer[cursor]
                    row_epoch = wp.atomic_add(next_executable, interaction_index, wp.int32(0))
                    if row_epoch == epoch:
                        if _nodes_ready(
                            nodes,
                            node_count,
                            node_start,
                            node_stride,
                            per_node_done,
                            interaction_index,
                            epoch,
                        ):
                            if _nodes_ready(
                                nodes,
                                node_count,
                                node_start,
                                node_stride,
                                per_node_done,
                                interaction_index,
                                epoch,
                            ):
                                old = wp.atomic_cas(next_executable, interaction_index, epoch, epoch + wp.int32(1))
                                if old == epoch:
                                    seed = interaction_index + epoch * wp.int32(65537)
                                    value = _descriptor_payload(
                                        work_iters,
                                        seed,
                                        family[interaction_index],
                                        row_size[interaction_index],
                                        projection[interaction_index],
                                        max_rows,
                                        payload_mode,
                                    )
                                    wp.atomic_add(sink, 0, value)

                                    n = node_count[interaction_index]
                                    slot = wp.int32(0)
                                    while slot < n:
                                        node_id = nodes[slot, interaction_index]
                                        wp.atomic_add(per_node_done, node_id, wp.int32(1))
                                        slot = slot + wp.int32(1)
                                    wp.atomic_add(window_color_done, color_epoch, wp.int32(1))
                                    wp.atomic_add(world_done, world_id, wp.int32(1))
                    cursor = cursor + threads_per_world
            window_offset = window_offset + wp.int32(1)

        pass_count = pass_count + wp.int32(1)
        done = wp.atomic_add(world_done, world_id, wp.int32(0))

    if local_tid == wp.int32(0) and done < target_total:
        failed[0] = wp.int32(1)


@wp.kernel(enable_backward=False)
def _reset_color_chunk_scheduler_kernel(
    chunk_phase: wp.array[wp.int32],
    chunk_color: wp.array[wp.int32],
    chunk_next: wp.array[wp.int32],
    chunk_done: wp.array[wp.int32],
    chunk_busy: wp.array[wp.int32],
    chunk_done_worlds: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    sink: wp.array[wp.float32],
    num_worlds: wp.int32,
    total_color_epochs: wp.int32,
):
    tid = wp.tid()
    if tid < num_worlds:
        chunk_phase[tid] = wp.int32(0)
        chunk_color[tid] = wp.int32(0)
    if tid < total_color_epochs:
        chunk_next[tid] = wp.int32(0)
        chunk_done[tid] = wp.int32(0)
        chunk_busy[tid] = wp.int32(0)
    if tid == wp.int32(0):
        chunk_done_worlds[0] = wp.int32(0)
        total_done[0] = wp.int32(0)
        failed[0] = wp.int32(0)
        sink[0] = wp.float32(0.0)


@wp.kernel(enable_backward=False)
def _world_color_chunk_kernel(
    family: wp.array[wp.int32],
    row_size: wp.array[wp.int32],
    projection: wp.array[wp.int32],
    color_buffer: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    world_color_starts: wp.array[wp.int32],
    chunk_phase: wp.array[wp.int32],
    chunk_color: wp.array[wp.int32],
    chunk_next: wp.array[wp.int32],
    chunk_done: wp.array[wp.int32],
    chunk_busy: wp.array[wp.int32],
    chunk_done_worlds: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    sink: wp.array[wp.float32],
    num_worlds: wp.int32,
    num_colors: wp.int32,
    num_iterations: wp.int32,
    total_threads: wp.int32,
    rows_per_chunk: wp.int32,
    work_iters: wp.int32,
    max_passes: wp.int32,
    max_rows: wp.int32,
    payload_mode: wp.int32,
):
    tid = wp.tid()
    pass_count = wp.int32(0)
    done_worlds = wp.atomic_add(chunk_done_worlds, 0, wp.int32(0))
    while done_worlds < num_worlds and pass_count < max_passes:
        scan = wp.int32(0)
        while scan < num_worlds:
            world_id = (tid + pass_count + scan) % num_worlds
            phase = wp.atomic_add(chunk_phase, world_id, wp.int32(0))
            if phase < num_iterations:
                color_begin = world_color_starts[world_id]
                color_end = world_color_starts[world_id + wp.int32(1)]
                local_color = wp.atomic_add(chunk_color, world_id, wp.int32(0))
                global_color = color_begin + local_color
                if global_color >= color_end:
                    next_phase = phase + wp.int32(1)
                    chunk_phase[world_id] = next_phase
                    chunk_color[world_id] = wp.int32(0)
                    if next_phase >= num_iterations:
                        wp.atomic_add(chunk_done_worlds, 0, wp.int32(1))
                else:
                    start = color_starts[global_color]
                    end = color_starts[global_color + wp.int32(1)]
                    count = end - start
                    chunks = (count + rows_per_chunk - wp.int32(1)) / rows_per_chunk
                    color_epoch = phase * num_colors + global_color
                    if chunks <= wp.int32(0):
                        old_busy = wp.atomic_cas(chunk_busy, color_epoch, wp.int32(0), wp.int32(1))
                        if old_busy == wp.int32(0):
                            next_color = local_color + wp.int32(1)
                            next_phase = phase
                            if color_begin + next_color >= color_end:
                                next_color = wp.int32(0)
                                next_phase = phase + wp.int32(1)
                            chunk_phase[world_id] = next_phase
                            chunk_color[world_id] = next_color
                            if next_phase >= num_iterations:
                                wp.atomic_add(chunk_done_worlds, 0, wp.int32(1))
                    else:
                        chunk = wp.atomic_add(chunk_next, color_epoch, wp.int32(1))
                        if chunk < chunks:
                            base = chunk * rows_per_chunk
                            local_row = wp.int32(0)
                            while local_row < rows_per_chunk and base + local_row < count:
                                interaction_index = color_buffer[start + base + local_row]
                                seed = interaction_index + phase * wp.int32(65537)
                                value = _descriptor_payload(
                                    work_iters,
                                    seed,
                                    family[interaction_index],
                                    row_size[interaction_index],
                                    projection[interaction_index],
                                    max_rows,
                                    payload_mode,
                                )
                                wp.atomic_add(sink, 0, value)
                                wp.atomic_add(total_done, 0, wp.int32(1))
                                local_row = local_row + wp.int32(1)

                            old_done = wp.atomic_add(chunk_done, color_epoch, wp.int32(1))
                            if old_done + wp.int32(1) == chunks:
                                old_busy = wp.atomic_cas(chunk_busy, color_epoch, wp.int32(0), wp.int32(1))
                                if old_busy == wp.int32(0):
                                    next_color = local_color + wp.int32(1)
                                    next_phase = phase
                                    if color_begin + next_color >= color_end:
                                        next_color = wp.int32(0)
                                        next_phase = phase + wp.int32(1)
                                    chunk_phase[world_id] = next_phase
                                    chunk_color[world_id] = next_color
                                    if next_phase >= num_iterations:
                                        wp.atomic_add(chunk_done_worlds, 0, wp.int32(1))
            scan = scan + wp.int32(1)
        pass_count = pass_count + wp.int32(1)
        done_worlds = wp.atomic_add(chunk_done_worlds, 0, wp.int32(0))

    if tid == wp.int32(0) and done_worlds < num_worlds:
        failed[0] = wp.int32(1)


def _unique(nodes: list[int]) -> list[int]:
    out: list[int] = []
    for node in nodes:
        if node >= 0 and node not in out:
            out.append(node)
    return out


def _make_interaction(nodes: list[int]) -> list[int]:
    unique = _unique(nodes)
    if len(unique) > MAX_NODES_PER_INTERACTION:
        raise ValueError("interaction exceeds MAX_NODES_PER_INTERACTION")
    return unique


def _physics_color_tail(num_colors: int, head_colors: int, head_batch: int) -> tuple[list[list[int]], list[int], int]:
    interactions: list[list[int]] = []
    color_ends: list[int] = []
    next_node = 1

    for _ in range(head_colors):
        for i in range(head_batch):
            if i == head_batch - 1:
                interactions.append(_make_interaction([0, next_node]))
                next_node += 1
            else:
                interactions.append(_make_interaction([next_node, next_node + 1]))
                next_node += 2
        color_ends.append(len(interactions))

    for _ in range(max(0, num_colors - head_colors)):
        interactions.append(_make_interaction([0, next_node]))
        next_node += 1
        color_ends.append(len(interactions))

    return interactions, color_ends, next_node


def _physics_hub(num_colors: int, head_colors: int, head_batch: int) -> tuple[list[list[int]], list[int], int]:
    interactions: list[list[int]] = []
    color_ends: list[int] = []
    next_node = 1

    for _ in range(head_colors):
        for _i in range(head_batch):
            interactions.append(_make_interaction([0, next_node]))
            next_node += 1
            color_ends.append(len(interactions))

    for _ in range(max(0, num_colors - head_colors)):
        interactions.append(_make_interaction([0, next_node]))
        next_node += 1
        color_ends.append(len(interactions))

    return interactions, color_ends, next_node


def _head_only(num_colors: int, head_batch: int) -> tuple[list[list[int]], list[int], int]:
    interactions: list[list[int]] = []
    color_ends: list[int] = []
    next_node = 0
    for _ in range(num_colors):
        for _i in range(head_batch):
            interactions.append(_make_interaction([next_node, next_node + 1]))
            next_node += 2
        color_ends.append(len(interactions))
    return interactions, color_ends, next_node


def _descriptor_arrays(family: np.ndarray, node_count: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_size = np.ones_like(family, dtype=np.int32)
    projection = np.zeros_like(family, dtype=np.int32)
    for i, fam_raw in enumerate(family):
        fam = int(fam_raw)
        n = int(node_count[i])
        if fam == 0:  # ADBS/revolute-like joint block.
            row_size[i] = 3
            projection[i] = 0
        elif fam == 1:  # Contact normal + two friction rows.
            row_size[i] = 3
            projection[i] = 2
        elif fam == 2:  # Cloth triangle.
            row_size[i] = 3
            projection[i] = 1
        elif fam == 3:  # Cloth bending hinge.
            row_size[i] = 1
            projection[i] = 1
        elif fam == 4:  # Soft tet / NH block.
            row_size[i] = 2
            projection[i] = 1
        elif fam == 5:  # Hex; expensive and wider.
            row_size[i] = 4
            projection[i] = 1
        else:
            row_size[i] = max(1, min(MAX_NODES_PER_INTERACTION, n))
            projection[i] = 1
    return family.astype(np.int32, copy=False), row_size, projection


def _synthetic_family(node_count: np.ndarray) -> np.ndarray:
    family = np.zeros_like(node_count, dtype=np.int32)
    for i, n_raw in enumerate(node_count):
        n = int(n_raw)
        if n <= 2:
            family[i] = 1 if i % 3 == 0 else 0
        elif n == 3:
            family[i] = 2
        elif n == 4:
            family[i] = 4
        else:
            family[i] = 5
    return family


def _build_graph(
    name: str,
    interactions: list[list[int]],
    color_ends: list[int],
    num_nodes: int,
    family_override: np.ndarray | None = None,
    world_starts_override: np.ndarray | None = None,
    world_color_starts_override: np.ndarray | None = None,
) -> SyntheticGraph:
    num_interactions = len(interactions)
    nodes = np.full((MAX_NODES_PER_INTERACTION, num_interactions), -1, dtype=np.int32)
    node_count = np.zeros(num_interactions, dtype=np.int32)
    for i, interaction_nodes in enumerate(interactions):
        node_count[i] = len(interaction_nodes)
        for slot, node_id in enumerate(interaction_nodes):
            nodes[slot, i] = node_id

    node_start = np.zeros((MAX_NODES_PER_INTERACTION, num_interactions), dtype=np.int32)
    node_stride = np.zeros((MAX_NODES_PER_INTERACTION, num_interactions), dtype=np.int32)
    per_node_counts = np.zeros(num_nodes, dtype=np.int32)
    color_buffer = np.arange(num_interactions, dtype=np.int32)
    color_starts = np.zeros(len(color_ends) + 1, dtype=np.int32)
    color_starts[1:] = np.asarray(color_ends, dtype=np.int32)
    if world_starts_override is None:
        world_starts = np.asarray([0, num_interactions], dtype=np.int32)
    else:
        world_starts = world_starts_override.astype(np.int32, copy=False)
    if world_color_starts_override is None:
        world_color_starts = np.asarray([0, len(color_ends)], dtype=np.int32)
    else:
        world_color_starts = world_color_starts_override.astype(np.int32, copy=False)

    color_begin = 0
    for color_end in color_ends:
        for cursor in range(color_begin, color_end):
            interaction_index = int(color_buffer[cursor])
            for slot, node_id in enumerate(interactions[interaction_index]):
                node_start[slot, interaction_index] = per_node_counts[node_id]
        for cursor in range(color_begin, color_end):
            interaction_index = int(color_buffer[cursor])
            for node_id in interactions[interaction_index]:
                per_node_counts[node_id] += 1
        color_begin = color_end

    for interaction_index, interaction_nodes in enumerate(interactions):
        for slot, node_id in enumerate(interaction_nodes):
            node_stride[slot, interaction_index] = per_node_counts[node_id]

    family_seed = (
        _synthetic_family(node_count) if family_override is None else family_override.astype(np.int32, copy=False)
    )
    family, row_size, projection = _descriptor_arrays(family_seed, node_count)

    return SyntheticGraph(
        name=name,
        nodes=nodes,
        node_count=node_count,
        node_start=node_start,
        node_stride=node_stride,
        world_starts=world_starts,
        world_color_starts=world_color_starts,
        family=family,
        row_size=row_size,
        projection=projection,
        color_buffer=color_buffer,
        color_starts=color_starts,
        num_nodes=num_nodes,
    )


def _color_ranges_from_world(world) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    if world.step_layout == "single_world":
        starts = world._partitioner.color_starts.numpy()
        num_colors = int(world._partitioner.num_colors.numpy()[0])
        for color in range(num_colors):
            ranges.append((int(starts[color]), int(starts[color + 1])))
        return ranges

    starts = world._world_color_starts.numpy()
    csr = world._world_csr_offsets.numpy()
    num_colors_per_world = world._world_num_colors.numpy()
    for world_id in range(world.num_worlds):
        base = int(csr[world_id])
        for color in range(int(num_colors_per_world[world_id])):
            ranges.append((base + int(starts[world_id, color]), base + int(starts[world_id, color + 1])))
    return ranges


def _element_ids_from_world(world) -> np.ndarray:
    if world.step_layout == "single_world":
        return world._partitioner.element_ids_by_color.numpy()
    return world._world_element_ids_by_color.numpy()


def make_graph_from_world(world, name: str) -> SyntheticGraph:
    active = int(world._num_active_constraints.numpy()[0])
    elements = world._elements.numpy()
    element_family = world._element_family.numpy()
    eids = _element_ids_from_world(world)

    interactions: list[list[int]] = []
    families: list[int] = []
    color_ends: list[int] = []
    world_starts: list[int] = [0]
    world_color_starts: list[int] = [0]
    max_node = -1

    if world.step_layout == "single_world":
        color_ranges = [_color_ranges_from_world(world)]
    else:
        starts = world._world_color_starts.numpy()
        csr = world._world_csr_offsets.numpy()
        num_colors_per_world = world._world_num_colors.numpy()
        color_ranges = []
        for world_id in range(world.num_worlds):
            base = int(csr[world_id])
            ranges: list[tuple[int, int]] = []
            for color in range(int(num_colors_per_world[world_id])):
                ranges.append((base + int(starts[world_id, color]), base + int(starts[world_id, color + 1])))
            color_ranges.append(ranges)

    for ranges in color_ranges:
        for start, end in ranges:
            for cursor in range(start, end):
                eid = int(eids[cursor])
                if eid < 0 or eid >= active:
                    continue
                bodies = elements[eid]["bodies"]
                nodes = [int(node) for node in bodies if int(node) >= 0]
                if not nodes:
                    continue
                interactions.append(_make_interaction(nodes))
                families.append(int(element_family[eid]))
                max_node = max(max_node, *nodes)
            color_ends.append(len(interactions))
        world_starts.append(len(interactions))
        world_color_starts.append(len(color_ends))

    if not interactions:
        interactions = [_make_interaction([0])]
        families = [0]
        color_ends = [1]
        world_starts = [0, 1]
        world_color_starts = [0, 1]
        max_node = 0

    return _build_graph(
        name,
        interactions,
        color_ends,
        max_node + 1,
        family_override=np.asarray(families, dtype=np.int32),
        world_starts_override=np.asarray(world_starts, dtype=np.int32),
        world_color_starts_override=np.asarray(world_color_starts, dtype=np.int32),
    )


def build_real_scene_graph(scene: str, args: argparse.Namespace) -> SyntheticGraph:
    if scene == "h1_1":
        handle = h1_flat.build(1, "phoenx", args.substeps, args.solver_iterations)
    elif scene == "h1_64":
        handle = h1_flat.build(64, "phoenx", args.substeps, args.solver_iterations)
    elif scene == "g1_64":
        handle = g1_flat.build(64, "phoenx", args.substeps, args.solver_iterations)
    elif scene == "tower_1":
        handle = tower.build(
            num_worlds=1,
            solver_name="phoenx",
            substeps=args.substeps,
            solver_iterations=args.solver_iterations,
            step_layout="single_world",
        )
    elif scene == "tower_32":
        handle = tower.build(
            num_worlds=32,
            solver_name="phoenx",
            substeps=args.substeps,
            solver_iterations=args.solver_iterations,
            step_layout="multi_world",
        )
    else:
        raise ValueError(f"unknown real scene: {scene}")

    for _ in range(args.scene_prime_frames):
        handle.simulate_one_frame()
    wp.synchronize_device()
    world = _extract_solver(handle).world
    return make_graph_from_world(world, scene)


def make_graph(name: str, num_colors: int, head_colors: int, head_batch: int) -> SyntheticGraph:
    if name == "tail":
        interactions, color_ends, num_nodes = _physics_color_tail(num_colors, head_colors, head_batch)
    elif name == "hub":
        interactions, color_ends, num_nodes = _physics_hub(num_colors, head_colors, head_batch)
    elif name == "head":
        interactions, color_ends, num_nodes = _head_only(num_colors, head_batch)
    else:
        raise ValueError(f"unknown graph: {name}")
    return _build_graph(name, interactions, color_ends, num_nodes)


@dataclass
class DeviceGraph:
    graph: SyntheticGraph
    nodes: wp.array
    node_count: wp.array
    node_start: wp.array
    node_stride: wp.array
    world_starts: wp.array
    world_color_starts: wp.array
    family: wp.array
    row_size: wp.array
    projection: wp.array
    color_buffer: wp.array
    color_starts: wp.array
    per_node_done: wp.array
    next_executable: wp.array
    world_done: wp.array
    total_done: wp.array
    failed: wp.array
    sink: wp.array
    window_phase: wp.array
    window_color: wp.array
    window_color_done: wp.array
    chunk_phase: wp.array
    chunk_color: wp.array
    chunk_next: wp.array
    chunk_done: wp.array
    chunk_busy: wp.array
    chunk_done_worlds: wp.array


def upload_graph(graph: SyntheticGraph, device: wp.context.Devicelike, *, iterations: int = 1) -> DeviceGraph:
    num_interactions = int(graph.node_count.shape[0])
    total_color_epochs = max(1, (len(graph.color_starts) - 1) * max(1, int(iterations)))
    return DeviceGraph(
        graph=graph,
        nodes=wp.array(graph.nodes, dtype=wp.int32, device=device),
        node_count=wp.array(graph.node_count, dtype=wp.int32, device=device),
        node_start=wp.array(graph.node_start, dtype=wp.int32, device=device),
        node_stride=wp.array(graph.node_stride, dtype=wp.int32, device=device),
        world_starts=wp.array(graph.world_starts, dtype=wp.int32, device=device),
        world_color_starts=wp.array(graph.world_color_starts, dtype=wp.int32, device=device),
        family=wp.array(graph.family, dtype=wp.int32, device=device),
        row_size=wp.array(graph.row_size, dtype=wp.int32, device=device),
        projection=wp.array(graph.projection, dtype=wp.int32, device=device),
        color_buffer=wp.array(graph.color_buffer, dtype=wp.int32, device=device),
        color_starts=wp.array(graph.color_starts, dtype=wp.int32, device=device),
        per_node_done=wp.zeros(max(1, graph.num_nodes), dtype=wp.int32, device=device),
        next_executable=wp.zeros(max(1, num_interactions), dtype=wp.int32, device=device),
        world_done=wp.zeros(max(1, graph.world_starts.shape[0] - 1), dtype=wp.int32, device=device),
        total_done=wp.zeros(1, dtype=wp.int32, device=device),
        failed=wp.zeros(1, dtype=wp.int32, device=device),
        sink=wp.zeros(1, dtype=wp.float32, device=device),
        window_phase=wp.zeros(max(1, graph.world_starts.shape[0] - 1), dtype=wp.int32, device=device),
        window_color=wp.zeros(max(1, graph.world_starts.shape[0] - 1), dtype=wp.int32, device=device),
        window_color_done=wp.zeros(total_color_epochs, dtype=wp.int32, device=device),
        chunk_phase=wp.zeros(max(1, graph.world_starts.shape[0] - 1), dtype=wp.int32, device=device),
        chunk_color=wp.zeros(max(1, graph.world_starts.shape[0] - 1), dtype=wp.int32, device=device),
        chunk_next=wp.zeros(total_color_epochs, dtype=wp.int32, device=device),
        chunk_done=wp.zeros(total_color_epochs, dtype=wp.int32, device=device),
        chunk_busy=wp.zeros(total_color_epochs, dtype=wp.int32, device=device),
        chunk_done_worlds=wp.zeros(1, dtype=wp.int32, device=device),
    )


def _launch_reset(dg: DeviceGraph, device: wp.context.Devicelike) -> None:
    dim = max(dg.graph.num_nodes, int(dg.graph.node_count.shape[0]), int(dg.graph.world_starts.shape[0] - 1), 1)
    wp.launch(
        _reset_scheduler_kernel,
        dim=dim,
        inputs=[
            dg.per_node_done,
            dg.next_executable,
            dg.world_done,
            dg.total_done,
            dg.failed,
            dg.sink,
            wp.int32(dg.graph.num_nodes),
            wp.int32(dg.graph.node_count.shape[0]),
            wp.int32(dg.graph.world_starts.shape[0] - 1),
        ],
        device=device,
    )


def make_colored_runner(
    dg: DeviceGraph,
    *,
    iterations: int,
    work_iters: int,
    max_rows: int,
    payload_mode: int,
    device: wp.context.Devicelike,
):
    color_starts = dg.graph.color_starts

    def run() -> None:
        _launch_reset(dg, device)
        for _epoch in range(iterations):
            for color_index in range(len(color_starts) - 1):
                start = int(color_starts[color_index])
                end = int(color_starts[color_index + 1])
                count = end - start
                if count <= 0:
                    continue
                wp.launch(
                    _colored_color_kernel,
                    dim=count,
                    inputs=[
                        dg.nodes,
                        dg.node_count,
                        dg.family,
                        dg.row_size,
                        dg.projection,
                        dg.color_buffer,
                        dg.per_node_done,
                        dg.total_done,
                        dg.sink,
                        wp.int32(start),
                        wp.int32(work_iters),
                        wp.int32(max_rows),
                        wp.int32(payload_mode),
                    ],
                    device=device,
                )

    return run


def make_overlap_colored_runner(
    dg: DeviceGraph,
    *,
    iterations: int,
    work_iters: int,
    num_streams: int,
    max_spin_passes: int,
    max_rows: int,
    payload_mode: int,
    device: wp.context.Devicelike,
):
    color_starts = dg.graph.color_starts
    streams = [wp.Stream(device) for _ in range(max(1, int(num_streams)))]

    def run() -> None:
        main_stream = wp.get_stream(device)
        _launch_reset(dg, device)
        for stream in streams:
            with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                wp.wait_stream(main_stream)
        for epoch in range(iterations):
            for color_index in range(len(color_starts) - 1):
                start = int(color_starts[color_index])
                end = int(color_starts[color_index + 1])
                count = end - start
                if count <= 0:
                    continue
                stream = streams[color_index % len(streams)]
                with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                    wp.launch(
                        _overlap_color_kernel,
                        dim=count,
                        inputs=[
                            dg.nodes,
                            dg.node_count,
                            dg.node_start,
                            dg.node_stride,
                            dg.family,
                            dg.row_size,
                            dg.projection,
                            dg.color_buffer,
                            dg.per_node_done,
                            dg.next_executable,
                            dg.total_done,
                            dg.failed,
                            dg.sink,
                            wp.int32(start),
                            wp.int32(epoch),
                            wp.int32(work_iters),
                            wp.int32(max_spin_passes),
                            wp.int32(max_rows),
                            wp.int32(payload_mode),
                        ],
                        device=device,
                    )
        for stream in streams:
            wp.wait_stream(stream)

    return run


def make_node_ready_runner(
    dg: DeviceGraph,
    *,
    iterations: int,
    work_iters: int,
    total_threads: int,
    max_passes: int,
    max_rows: int,
    payload_mode: int,
    device: wp.context.Devicelike,
):
    def run() -> None:
        _launch_reset(dg, device)
        wp.launch(
            _node_ready_scan_kernel,
            dim=total_threads,
            inputs=[
                dg.nodes,
                dg.node_count,
                dg.node_start,
                dg.node_stride,
                dg.family,
                dg.row_size,
                dg.projection,
                dg.per_node_done,
                dg.next_executable,
                dg.total_done,
                dg.failed,
                dg.sink,
                wp.int32(dg.graph.node_count.shape[0]),
                wp.int32(iterations),
                wp.int32(total_threads),
                wp.int32(work_iters),
                wp.int32(max_passes),
                wp.int32(max_rows),
                wp.int32(payload_mode),
            ],
            device=device,
        )

    return run


def make_world_ready_runner(
    dg: DeviceGraph,
    *,
    iterations: int,
    work_iters: int,
    threads_per_world: int,
    max_passes: int,
    max_rows: int,
    payload_mode: int,
    device: wp.context.Devicelike,
):
    def run() -> None:
        _launch_reset(dg, device)
        wp.launch(
            _world_ready_scan_kernel,
            dim=max(1, (dg.graph.world_starts.shape[0] - 1) * threads_per_world),
            inputs=[
                dg.nodes,
                dg.node_count,
                dg.node_start,
                dg.node_stride,
                dg.family,
                dg.row_size,
                dg.projection,
                dg.world_starts,
                dg.per_node_done,
                dg.next_executable,
                dg.world_done,
                dg.failed,
                dg.sink,
                wp.int32(dg.graph.world_starts.shape[0] - 1),
                wp.int32(iterations),
                wp.int32(threads_per_world),
                wp.int32(work_iters),
                wp.int32(max_passes),
                wp.int32(max_rows),
                wp.int32(payload_mode),
            ],
            device=device,
        )

    return run


def make_world_window_ready_runner(
    dg: DeviceGraph,
    *,
    iterations: int,
    work_iters: int,
    threads_per_world: int,
    window_colors: int,
    max_passes: int,
    max_rows: int,
    payload_mode: int,
    device: wp.context.Devicelike,
):
    num_worlds = int(dg.graph.world_starts.shape[0] - 1)
    num_colors = len(dg.graph.color_starts) - 1
    total_color_epochs = max(1, num_colors * max(1, int(iterations)))

    def run() -> None:
        wp.launch(
            _reset_window_scheduler_kernel,
            dim=max(dg.graph.num_nodes, int(dg.graph.node_count.shape[0]), num_worlds, total_color_epochs, 1),
            inputs=[
                dg.per_node_done,
                dg.next_executable,
                dg.world_done,
                dg.window_phase,
                dg.window_color,
                dg.window_color_done,
                dg.failed,
                dg.sink,
                wp.int32(dg.graph.num_nodes),
                wp.int32(dg.graph.node_count.shape[0]),
                wp.int32(num_worlds),
                wp.int32(total_color_epochs),
            ],
            device=device,
        )
        wp.launch(
            _world_window_ready_kernel,
            dim=max(1, num_worlds * threads_per_world),
            inputs=[
                dg.nodes,
                dg.node_count,
                dg.node_start,
                dg.node_stride,
                dg.family,
                dg.row_size,
                dg.projection,
                dg.color_buffer,
                dg.color_starts,
                dg.world_color_starts,
                dg.per_node_done,
                dg.next_executable,
                dg.world_done,
                dg.window_phase,
                dg.window_color,
                dg.window_color_done,
                dg.failed,
                dg.sink,
                wp.int32(num_worlds),
                wp.int32(num_colors),
                wp.int32(iterations),
                wp.int32(threads_per_world),
                wp.int32(window_colors),
                wp.int32(work_iters),
                wp.int32(max_passes),
                wp.int32(max_rows),
                wp.int32(payload_mode),
            ],
            device=device,
        )

    return run


def make_world_color_chunk_runner(
    dg: DeviceGraph,
    *,
    iterations: int,
    work_iters: int,
    total_threads: int,
    rows_per_chunk: int,
    max_passes: int,
    max_rows: int,
    payload_mode: int,
    device: wp.context.Devicelike,
):
    num_worlds = int(dg.graph.world_starts.shape[0] - 1)

    def run() -> None:
        wp.launch(
            _reset_color_chunk_scheduler_kernel,
            dim=max(1, num_worlds, (len(dg.graph.color_starts) - 1) * iterations),
            inputs=[
                dg.chunk_phase,
                dg.chunk_color,
                dg.chunk_next,
                dg.chunk_done,
                dg.chunk_busy,
                dg.chunk_done_worlds,
                dg.total_done,
                dg.failed,
                dg.sink,
                wp.int32(num_worlds),
                wp.int32((len(dg.graph.color_starts) - 1) * iterations),
            ],
            device=device,
        )
        wp.launch(
            _world_color_chunk_kernel,
            dim=total_threads,
            inputs=[
                dg.family,
                dg.row_size,
                dg.projection,
                dg.color_buffer,
                dg.color_starts,
                dg.world_color_starts,
                dg.chunk_phase,
                dg.chunk_color,
                dg.chunk_next,
                dg.chunk_done,
                dg.chunk_busy,
                dg.chunk_done_worlds,
                dg.total_done,
                dg.failed,
                dg.sink,
                wp.int32(num_worlds),
                wp.int32(len(dg.graph.color_starts) - 1),
                wp.int32(iterations),
                wp.int32(total_threads),
                wp.int32(rows_per_chunk),
                wp.int32(work_iters),
                wp.int32(max_passes),
                wp.int32(max_rows),
                wp.int32(payload_mode),
            ],
            device=device,
        )

    return run


def validate_runner(label: str, dg: DeviceGraph, run, expected_total: int, *, use_world_done: bool = False) -> None:
    run()
    wp.synchronize_device()
    if use_world_done:
        total = int(np.sum(dg.world_done.numpy()))
    else:
        total = int(dg.total_done.numpy()[0])
    failed = int(dg.failed.numpy()[0])
    if total != expected_total:
        raise RuntimeError(f"{label} failed: total_done={total}, failed={failed}, expected={expected_total}")


def _payload_modes(arg: str) -> tuple[tuple[str, int], ...]:
    if arg == "specialized":
        return (("specialized", PAYLOAD_SPECIALIZED),)
    if arg == "unified":
        return (("unified", PAYLOAD_UNIFIED),)
    return (("specialized", PAYLOAD_SPECIALIZED), ("unified", PAYLOAD_UNIFIED))


def _print_graph_stats(graph: SyntheticGraph) -> None:
    fam_values, fam_counts = np.unique(graph.family, return_counts=True)
    fam_text = ", ".join(f"{int(f)}:{int(c)}" for f, c in zip(fam_values, fam_counts, strict=True))
    color_counts = np.diff(graph.color_starts)
    print(
        f"\n=== {graph.name}: {graph.node_count.shape[0]} interactions, "
        f"{len(graph.color_starts) - 1} colors, {graph.world_starts.shape[0] - 1} worlds, {graph.num_nodes} nodes ==="
    )
    print(
        f"families={{{fam_text}}} row_size[min/median/max]="
        f"{int(graph.row_size.min())}/{int(np.median(graph.row_size))}/{int(graph.row_size.max())} "
        f"color_size[min/median/max]={int(color_counts.min())}/{int(np.median(color_counts))}/{int(color_counts.max())}"
    )


def run_graph_case(args: argparse.Namespace, graph: SyntheticGraph) -> None:
    device = wp.get_device(args.device)
    expected_total = int(graph.node_count.shape[0]) * int(args.iterations)
    max_rows = int(args.max_rows) if int(args.max_rows) > 0 else int(graph.row_size.max())
    _print_graph_stats(graph)

    colored_kernel_nodes = int(args.iterations) * (len(graph.color_starts) - 1)
    for payload_label, payload_mode in _payload_modes(args.payload_mode):
        colored = upload_graph(graph, device, iterations=args.iterations)
        colored_runner = make_colored_runner(
            colored,
            iterations=args.iterations,
            work_iters=args.work_iters,
            max_rows=max_rows,
            payload_mode=payload_mode,
            device=device,
        )
        validate_runner(f"colored/{payload_label}", colored, colored_runner, expected_total)
        colored_min, colored_med = _bench(colored_runner, args.n_runs, args.warmup, args.trials)

        overlap_colored = upload_graph(graph, device, iterations=args.iterations)
        overlap_colored_runner = make_overlap_colored_runner(
            overlap_colored,
            iterations=args.iterations,
            work_iters=args.work_iters,
            num_streams=args.overlap_streams,
            max_spin_passes=args.overlap_spin_passes,
            max_rows=max_rows,
            payload_mode=payload_mode,
            device=device,
        )
        validate_runner(f"overlap/{payload_label}", overlap_colored, overlap_colored_runner, expected_total)
        overlap_min, overlap_med = _bench(overlap_colored_runner, args.n_runs, args.warmup, args.trials)

        node_ready = upload_graph(graph, device, iterations=args.iterations)
        node_ready_runner = make_node_ready_runner(
            node_ready,
            iterations=args.iterations,
            work_iters=args.work_iters,
            total_threads=args.total_threads,
            max_passes=args.max_passes,
            max_rows=max_rows,
            payload_mode=payload_mode,
            device=device,
        )
        validate_runner(f"node_ready/{payload_label}", node_ready, node_ready_runner, expected_total)
        node_min, node_med = _bench(node_ready_runner, args.n_runs, args.warmup, args.trials)

        world_ready = upload_graph(graph, device, iterations=args.iterations)
        world_ready_runner = make_world_ready_runner(
            world_ready,
            iterations=args.iterations,
            work_iters=args.work_iters,
            threads_per_world=args.world_tpw,
            max_passes=args.max_passes,
            max_rows=max_rows,
            payload_mode=payload_mode,
            device=device,
        )
        validate_runner(
            f"world_ready/{payload_label}", world_ready, world_ready_runner, expected_total, use_world_done=True
        )
        world_min, world_med = _bench(world_ready_runner, args.n_runs, args.warmup, args.trials)

        window_ready = upload_graph(graph, device, iterations=args.iterations)
        window_ready_runner = make_world_window_ready_runner(
            window_ready,
            iterations=args.iterations,
            work_iters=args.work_iters,
            threads_per_world=args.world_tpw,
            window_colors=args.window_colors,
            max_passes=args.max_passes,
            max_rows=max_rows,
            payload_mode=payload_mode,
            device=device,
        )
        validate_runner(
            f"window_ready/{payload_label}", window_ready, window_ready_runner, expected_total, use_world_done=True
        )
        window_min, window_med = _bench(window_ready_runner, args.n_runs, args.warmup, args.trials)

        color_chunk = upload_graph(graph, device, iterations=args.iterations)
        color_chunk_runner = make_world_color_chunk_runner(
            color_chunk,
            iterations=args.iterations,
            work_iters=args.work_iters,
            total_threads=args.total_threads,
            rows_per_chunk=args.rows_per_chunk,
            max_passes=args.max_passes,
            max_rows=max_rows,
            payload_mode=payload_mode,
            device=device,
        )
        validate_runner(f"color_chunk/{payload_label}", color_chunk, color_chunk_runner, expected_total)
        color_chunk_min, color_chunk_med = _bench(color_chunk_runner, args.n_runs, args.warmup, args.trials)

        print(
            f"{payload_label:11s} colored    min={colored_min:9.3f} ms  med={colored_med:9.3f} ms  "
            f"kernel_nodes/frame={colored_kernel_nodes}"
        )
        print(
            f"{payload_label:11s} overlap   min={overlap_min:9.3f} ms  med={overlap_med:9.3f} ms  "
            f"streams={args.overlap_streams}"
        )
        print(
            f"{payload_label:11s} node_ready min={node_min:9.3f} ms  med={node_med:9.3f} ms  "
            f"threads={args.total_threads}"
        )
        print(f"{payload_label:11s} world_ready min={world_min:8.3f} ms  med={world_med:9.3f} ms  tpw={args.world_tpw}")
        print(
            f"{payload_label:11s} window     min={window_min:8.3f} ms  "
            f"med={window_med:9.3f} ms  tpw={args.world_tpw} window={args.window_colors}"
        )
        print(
            f"{payload_label:11s} color_chunk min={color_chunk_min:8.3f} ms  "
            f"med={color_chunk_med:9.3f} ms  threads={args.total_threads} rows/chunk={args.rows_per_chunk}"
        )
        overlap_speed = colored_min / overlap_min if overlap_min > 0.0 else 0.0
        print(f"{payload_label:11s} overlap relative to colored: {overlap_speed:6.3f}x")
        speed = colored_min / node_min if node_min > 0.0 else 0.0
        print(f"{payload_label:11s} node_ready relative to colored: {speed:6.3f}x")
        world_speed = colored_min / world_min if world_min > 0.0 else 0.0
        print(f"{payload_label:11s} world_ready relative to colored: {world_speed:6.3f}x")
        window_speed = colored_min / window_min if window_min > 0.0 else 0.0
        print(f"{payload_label:11s} window relative to colored: {window_speed:6.3f}x")
        color_chunk_speed = colored_min / color_chunk_min if color_chunk_min > 0.0 else 0.0
        print(f"{payload_label:11s} color_chunk relative to colored: {color_chunk_speed:6.3f}x")


def run_case(args: argparse.Namespace, graph_name: str) -> None:
    run_graph_case(args, make_graph(graph_name, args.colors, args.head_colors, args.head_batch))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph", choices=("none", "head", "tail", "hub", "all"), default="all")
    parser.add_argument(
        "--real-scenes",
        nargs="*",
        default=[],
        choices=("h1_1", "h1_64", "g1_64", "tower_1", "tower_32"),
        help="Optional real PhoenX scenes to extract into descriptor graphs.",
    )
    parser.add_argument("--colors", type=int, default=50)
    parser.add_argument("--head-colors", type=int, default=10)
    parser.add_argument("--head-batch", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--work-iters", type=int, default=64)
    parser.add_argument(
        "--max-rows", type=int, default=0, help="Unified payload row width. 0 uses each graph's max row size."
    )
    parser.add_argument(
        "--payload-mode",
        choices=("specialized", "unified", "both"),
        default="both",
        help="Compare divergent specialized payload, fixed-shape unified payload, or both.",
    )
    parser.add_argument("--total-threads", type=int, default=1024)
    parser.add_argument("--overlap-streams", type=int, default=4)
    parser.add_argument("--overlap-spin-passes", type=int, default=65536)
    parser.add_argument("--world-tpw", type=int, default=32)
    parser.add_argument("--window-colors", type=int, default=4)
    parser.add_argument("--rows-per-chunk", type=int, default=4)
    parser.add_argument("--max-passes", type=int, default=65536)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--scene-prime-frames", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    graph_names = () if args.graph == "none" else (("head", "tail", "hub") if args.graph == "all" else (args.graph,))
    print(
        f"device={args.device} iterations={args.iterations} work_iters={args.work_iters} "
        f"payload={args.payload_mode} n_runs={args.n_runs} trials={args.trials}"
    )
    for graph_name in graph_names:
        run_case(args, graph_name)
    for scene in args.real_scenes:
        run_graph_case(args, build_real_scene_graph(scene, args))


if __name__ == "__main__":
    main()
