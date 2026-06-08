# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Synthetic PhoenX megakernel scheduling benchmark.

This is intentionally not a solver dispatch path. It compares the scheduling
shape from ``SchedulingExperiments.Cuda`` against a conventional colored PGS
loop on synthetic interaction graphs:

* ``colored`` runs one captured kernel node per color and epoch. This is a
  naive launch-bound baseline, not PhoenX's production fast-tail path;
* ``node_ready`` runs one persistent scan kernel that uses per-node epoch
  counters and atomics to decide when each interaction may execute.

The payload is a configurable arithmetic loop, not PhoenX constraint math. The
goal is to measure whether megakernel scheduling overhead is plausible before
we wire it into real joint/contact rows.

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.bench_megakernel_scheduling
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.bench_threads_per_world import _bench

MAX_NODES_PER_INTERACTION = 8


@dataclass(frozen=True)
class SyntheticGraph:
    name: str
    nodes: np.ndarray
    node_count: np.ndarray
    node_start: np.ndarray
    node_stride: np.ndarray
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
    total_done: wp.array[wp.int32],
    failed: wp.array[wp.int32],
    sink: wp.array[wp.float32],
    num_nodes: wp.int32,
    num_interactions: wp.int32,
):
    tid = wp.tid()
    if tid < num_nodes:
        per_node_done[tid] = wp.int32(0)
    if tid < num_interactions:
        next_executable[tid] = wp.int32(0)
    if tid == wp.int32(0):
        total_done[0] = wp.int32(0)
        failed[0] = wp.int32(0)
        sink[0] = wp.float32(0.0)


@wp.kernel(enable_backward=False)
def _colored_color_kernel(
    nodes: wp.array2d[wp.int32],
    node_count: wp.array[wp.int32],
    color_buffer: wp.array[wp.int32],
    per_node_done: wp.array[wp.int32],
    total_done: wp.array[wp.int32],
    sink: wp.array[wp.float32],
    color_start: wp.int32,
    work_iters: wp.int32,
):
    lane = wp.tid()
    interaction_index = color_buffer[color_start + lane]
    node_count_i = node_count[interaction_index]
    seed = interaction_index + total_done[0] * wp.int32(17)
    value = _simulate_payload(work_iters, seed)
    wp.atomic_add(sink, 0, value)

    slot = wp.int32(0)
    while slot < node_count_i:
        node_id = nodes[slot, interaction_index]
        per_node_done[node_id] = per_node_done[node_id] + wp.int32(1)
        slot = slot + wp.int32(1)
    wp.atomic_add(total_done, 0, wp.int32(1))


@wp.kernel(enable_backward=False)
def _node_ready_scan_kernel(
    nodes: wp.array2d[wp.int32],
    node_count: wp.array[wp.int32],
    node_start: wp.array2d[wp.int32],
    node_stride: wp.array2d[wp.int32],
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
                            value = _simulate_payload(work_iters, seed)
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


def _build_graph(
    name: str,
    interactions: list[list[int]],
    color_ends: list[int],
    num_nodes: int,
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

    return SyntheticGraph(
        name=name,
        nodes=nodes,
        node_count=node_count,
        node_start=node_start,
        node_stride=node_stride,
        color_buffer=color_buffer,
        color_starts=color_starts,
        num_nodes=num_nodes,
    )


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
    color_buffer: wp.array
    per_node_done: wp.array
    next_executable: wp.array
    total_done: wp.array
    failed: wp.array
    sink: wp.array


def upload_graph(graph: SyntheticGraph, device: wp.context.Devicelike) -> DeviceGraph:
    num_interactions = int(graph.node_count.shape[0])
    return DeviceGraph(
        graph=graph,
        nodes=wp.array(graph.nodes, dtype=wp.int32, device=device),
        node_count=wp.array(graph.node_count, dtype=wp.int32, device=device),
        node_start=wp.array(graph.node_start, dtype=wp.int32, device=device),
        node_stride=wp.array(graph.node_stride, dtype=wp.int32, device=device),
        color_buffer=wp.array(graph.color_buffer, dtype=wp.int32, device=device),
        per_node_done=wp.zeros(max(1, graph.num_nodes), dtype=wp.int32, device=device),
        next_executable=wp.zeros(max(1, num_interactions), dtype=wp.int32, device=device),
        total_done=wp.zeros(1, dtype=wp.int32, device=device),
        failed=wp.zeros(1, dtype=wp.int32, device=device),
        sink=wp.zeros(1, dtype=wp.float32, device=device),
    )


def _launch_reset(dg: DeviceGraph, device: wp.context.Devicelike) -> None:
    dim = max(dg.graph.num_nodes, int(dg.graph.node_count.shape[0]), 1)
    wp.launch(
        _reset_scheduler_kernel,
        dim=dim,
        inputs=[
            dg.per_node_done,
            dg.next_executable,
            dg.total_done,
            dg.failed,
            dg.sink,
            wp.int32(dg.graph.num_nodes),
            wp.int32(dg.graph.node_count.shape[0]),
        ],
        device=device,
    )


def make_colored_runner(
    dg: DeviceGraph,
    *,
    iterations: int,
    work_iters: int,
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
                        dg.color_buffer,
                        dg.per_node_done,
                        dg.total_done,
                        dg.sink,
                        wp.int32(start),
                        wp.int32(work_iters),
                    ],
                    device=device,
                )

    return run


def make_node_ready_runner(
    dg: DeviceGraph,
    *,
    iterations: int,
    work_iters: int,
    total_threads: int,
    max_passes: int,
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
            ],
            device=device,
        )

    return run


def validate_runner(label: str, dg: DeviceGraph, run, expected_total: int) -> None:
    run()
    wp.synchronize_device()
    total = int(dg.total_done.numpy()[0])
    failed = int(dg.failed.numpy()[0])
    if total != expected_total:
        raise RuntimeError(f"{label} failed: total_done={total}, failed={failed}, expected={expected_total}")


def run_case(args: argparse.Namespace, graph_name: str) -> None:
    device = wp.get_device(args.device)
    graph = make_graph(graph_name, args.colors, args.head_colors, args.head_batch)
    expected_total = int(graph.node_count.shape[0]) * int(args.iterations)
    print(
        f"\n=== {graph.name}: {graph.node_count.shape[0]} interactions, "
        f"{len(graph.color_starts) - 1} colors, {graph.num_nodes} nodes ==="
    )

    colored = upload_graph(graph, device)
    colored_runner = make_colored_runner(
        colored,
        iterations=args.iterations,
        work_iters=args.work_iters,
        device=device,
    )
    validate_runner("colored", colored, colored_runner, expected_total)
    colored_min, colored_med = _bench(colored_runner, args.n_runs, args.warmup, args.trials)

    node_ready = upload_graph(graph, device)
    node_ready_runner = make_node_ready_runner(
        node_ready,
        iterations=args.iterations,
        work_iters=args.work_iters,
        total_threads=args.total_threads,
        max_passes=args.max_passes,
        device=device,
    )
    validate_runner("node_ready", node_ready, node_ready_runner, expected_total)
    node_min, node_med = _bench(node_ready_runner, args.n_runs, args.warmup, args.trials)

    colored_kernel_nodes = int(args.iterations) * (len(graph.color_starts) - 1)
    print(
        f"colored     min={colored_min:9.3f} ms  med={colored_med:9.3f} ms  kernel_nodes/frame={colored_kernel_nodes}"
    )
    print(f"node_ready  min={node_min:9.3f} ms  med={node_med:9.3f} ms  threads={args.total_threads}")
    speed = colored_min / node_min if node_min > 0.0 else 0.0
    print(f"node_ready relative to colored: {speed:6.3f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph", choices=("head", "tail", "hub", "all"), default="all")
    parser.add_argument("--colors", type=int, default=50)
    parser.add_argument("--head-colors", type=int, default=10)
    parser.add_argument("--head-batch", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--work-iters", type=int, default=64)
    parser.add_argument("--total-threads", type=int, default=1024)
    parser.add_argument("--max-passes", type=int, default=65536)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    graph_names = ("head", "tail", "hub") if args.graph == "all" else (args.graph,)
    print(
        f"device={args.device} iterations={args.iterations} work_iters={args.work_iters} "
        f"n_runs={args.n_runs} trials={args.trials}"
    )
    for graph_name in graph_names:
        run_case(args, graph_name)


if __name__ == "__main__":
    main()
