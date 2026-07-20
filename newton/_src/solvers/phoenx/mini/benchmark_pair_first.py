# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Upper-bound benchmark for deterministic pair-first contact generation."""

from __future__ import annotations

import argparse
import json
import math
import time

import warp as wp


@wp.kernel(enable_backward=False, grid_stride=False)
def _prepare_pair_keys(keys: wp.array[wp.int32], values: wp.array[wp.int32], count: wp.int32, mask: wp.int32):
    i = wp.tid()
    keys[i] = (i * wp.int32(174763)) & mask
    values[i] = i


@wp.kernel(enable_backward=False, grid_stride=False)
def _prepare_contact_keys(
    keys: wp.array[wp.int32],
    values: wp.array[wp.int32],
    count: wp.int32,
    live_count: wp.int32,
    points_per_pair: wp.int32,
    pair_mask: wp.int32,
):
    i = wp.tid()
    if i < live_count:
        keys[i] = (i * wp.int32(174763)) & pair_mask
    else:
        keys[i] = pair_mask + wp.int32(1)
    values[i] = i


@wp.kernel(enable_backward=False, grid_stride=False)
def _gather_record(
    permutation: wp.array[wp.int32],
    src0: wp.array[wp.vec4],
    src1: wp.array[wp.vec4],
    src2: wp.array[wp.vec4],
    src3: wp.array[wp.vec4],
    src4: wp.array[wp.vec4],
    src5: wp.array[wp.vec4],
    dst0: wp.array[wp.vec4],
    dst1: wp.array[wp.vec4],
    dst2: wp.array[wp.vec4],
    dst3: wp.array[wp.vec4],
    dst4: wp.array[wp.vec4],
    dst5: wp.array[wp.vec4],
):
    i = wp.tid()
    source = permutation[i]
    dst0[i] = src0[source]
    dst1[i] = src1[source]
    dst2[i] = src2[source]
    dst3[i] = src3[source]
    dst4[i] = src4[source]
    dst5[i] = src5[source]


@wp.kernel(enable_backward=False, grid_stride=False)
def _gather_soa_record(
    permutation: wp.array[wp.int32],
    src_i0: wp.array[wp.int32],
    src_i1: wp.array[wp.int32],
    src_i2: wp.array[wp.int32],
    src_f0: wp.array[wp.float32],
    src_f1: wp.array[wp.float32],
    src_f2: wp.array[wp.float32],
    src_f3: wp.array[wp.float32],
    src_f4: wp.array[wp.float32],
    src_v0: wp.array[wp.vec3],
    src_v1: wp.array[wp.vec3],
    src_v2: wp.array[wp.vec3],
    src_v3: wp.array[wp.vec3],
    src_v4: wp.array[wp.vec3],
    dst_i0: wp.array[wp.int32],
    dst_i1: wp.array[wp.int32],
    dst_i2: wp.array[wp.int32],
    dst_f0: wp.array[wp.float32],
    dst_f1: wp.array[wp.float32],
    dst_f2: wp.array[wp.float32],
    dst_f3: wp.array[wp.float32],
    dst_f4: wp.array[wp.float32],
    dst_v0: wp.array[wp.vec3],
    dst_v1: wp.array[wp.vec3],
    dst_v2: wp.array[wp.vec3],
    dst_v3: wp.array[wp.vec3],
    dst_v4: wp.array[wp.vec3],
):
    i = wp.tid()
    source = permutation[i]
    dst_i0[i] = src_i0[source]
    dst_i1[i] = src_i1[source]
    dst_i2[i] = src_i2[source]
    dst_f0[i] = src_f0[source]
    dst_f1[i] = src_f1[source]
    dst_f2[i] = src_f2[source]
    dst_f3[i] = src_f3[source]
    dst_f4[i] = src_f4[source]
    dst_v0[i] = src_v0[source]
    dst_v1[i] = src_v1[source]
    dst_v2[i] = src_v2[source]
    dst_v3[i] = src_v3[source]
    dst_v4[i] = src_v4[source]


def _measure(graph: wp.Graph, replays: int, device: wp.Device) -> float:
    for _ in range(20):
        wp.capture_launch(graph)
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(replays):
        wp.capture_launch(graph)
    wp.synchronize_device(device)
    return (time.perf_counter() - start) * 1.0e6 / replays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=524_288)
    parser.add_argument("--points-per-pair", type=int, default=4)
    parser.add_argument("--replays", type=int, default=1000)
    args = parser.parse_args()

    if args.pairs <= 0 or args.pairs & (args.pairs - 1):
        raise ValueError("--pairs must be a positive power of two")
    if args.points_per_pair <= 0:
        raise ValueError("--points-per-pair must be positive")

    wp.init()
    device = wp.get_device("cuda:0")
    pair_count = args.pairs
    contact_capacity = pair_count * args.points_per_pair
    live_contacts = pair_count * 17 // 8
    pair_bits = int(math.log2(pair_count))
    contact_bits = int(math.ceil(math.log2(contact_capacity)))

    pair_keys = wp.empty(pair_count * 2, dtype=wp.int32, device=device)
    pair_values = wp.empty(pair_count * 2, dtype=wp.int32, device=device)
    contact_keys = wp.empty(contact_capacity * 2, dtype=wp.int32, device=device)
    contact_values = wp.empty(contact_capacity * 2, dtype=wp.int32, device=device)
    valid = wp.empty(contact_capacity, dtype=wp.int32, device=device)
    offsets = wp.empty(contact_capacity, dtype=wp.int32, device=device)
    src = [wp.zeros(contact_capacity, dtype=wp.vec4, device=device) for _ in range(6)]
    dst = [wp.empty(contact_capacity, dtype=wp.vec4, device=device) for _ in range(6)]
    soa_src_i = [wp.zeros(contact_capacity, dtype=wp.int32, device=device) for _ in range(3)]
    soa_dst_i = [wp.empty(contact_capacity, dtype=wp.int32, device=device) for _ in range(3)]
    soa_src_f = [wp.zeros(contact_capacity, dtype=wp.float32, device=device) for _ in range(5)]
    soa_dst_f = [wp.empty(contact_capacity, dtype=wp.float32, device=device) for _ in range(5)]
    soa_src_v = [wp.zeros(contact_capacity, dtype=wp.vec3, device=device) for _ in range(5)]
    soa_dst_v = [wp.empty(contact_capacity, dtype=wp.vec3, device=device) for _ in range(5)]

    with wp.ScopedCapture(device=device) as pair_capture:
        wp.launch(
            _prepare_pair_keys,
            dim=pair_count,
            inputs=[pair_keys, pair_values, pair_count, pair_count - 1],
            device=device,
        )
        wp.utils.radix_sort_pairs(pair_keys, pair_values, pair_count, end_bit=pair_bits)

    with wp.ScopedCapture(device=device) as contact_sort_capture:
        wp.launch(
            _prepare_contact_keys,
            dim=contact_capacity,
            inputs=[
                contact_keys,
                contact_values,
                contact_capacity,
                live_contacts,
                args.points_per_pair,
                contact_capacity - 1,
            ],
            device=device,
        )
        wp.utils.radix_sort_pairs(contact_keys, contact_values, contact_capacity, end_bit=contact_bits + 1)

    with wp.ScopedCapture(device=device) as current_capture:
        wp.launch(
            _prepare_contact_keys,
            dim=contact_capacity,
            inputs=[
                contact_keys,
                contact_values,
                contact_capacity,
                live_contacts,
                args.points_per_pair,
                contact_capacity - 1,
            ],
            device=device,
        )
        wp.utils.radix_sort_pairs(contact_keys, contact_values, contact_capacity, end_bit=contact_bits + 1)
        wp.launch(_gather_record, dim=live_contacts, inputs=[contact_values, *src], outputs=dst, device=device)

    with wp.ScopedCapture(device=device) as soa_gather_capture:
        wp.launch(
            _gather_soa_record,
            dim=live_contacts,
            inputs=[contact_values, *soa_src_i, *soa_src_f, *soa_src_v],
            outputs=[*soa_dst_i, *soa_dst_f, *soa_dst_v],
            device=device,
        )

    with wp.ScopedCapture(device=device) as packed_gather_capture:
        wp.launch(_gather_record, dim=live_contacts, inputs=[contact_values, *src], outputs=dst, device=device)

    with wp.ScopedCapture(device=device) as pair_compact_capture:
        wp.launch(
            _prepare_pair_keys,
            dim=pair_count,
            inputs=[pair_keys, pair_values, pair_count, pair_count - 1],
            device=device,
        )
        wp.utils.radix_sort_pairs(pair_keys, pair_values, pair_count, end_bit=pair_bits)
        wp.launch(_mark_pair_slots, dim=contact_capacity, inputs=[valid, args.points_per_pair], device=device)
        wp.utils.array_scan(valid, offsets, inclusive=False)
        wp.launch(_compact_record, dim=contact_capacity, inputs=[valid, offsets, *src], outputs=dst, device=device)

    pair_us = _measure(pair_capture.graph, args.replays, device)
    contact_sort_us = _measure(contact_sort_capture.graph, args.replays, device)
    current_us = _measure(current_capture.graph, args.replays, device)
    soa_gather_us = _measure(soa_gather_capture.graph, args.replays, device)
    packed_gather_us = _measure(packed_gather_capture.graph, args.replays, device)
    pair_compact_us = _measure(pair_compact_capture.graph, args.replays, device)
    print(
        json.dumps(
            {
                "pairs": pair_count,
                "contact_capacity": contact_capacity,
                "live_contacts": live_contacts,
                "record_bytes": 96,
                "pair_sort_us": pair_us,
                "contact_sort_us": contact_sort_us,
                "contact_sort_plus_gather_us": current_us,
                "soa_92b_gather_us": soa_gather_us,
                "packed_96b_gather_us": packed_gather_us,
                "packed_gather_speedup": soa_gather_us / packed_gather_us,
                "pair_sort_plus_variable_compaction_us": pair_compact_us,
                "optimistic_saved_us": current_us - pair_us,
                "optimistic_speedup": current_us / pair_us,
                "compacted_saved_us": current_us - pair_compact_us,
                "compacted_speedup": current_us / pair_compact_us,
            },
            indent=2,
        )
    )


@wp.kernel(enable_backward=False, grid_stride=False)
def _mark_pair_slots(valid: wp.array[wp.int32], points_per_pair: wp.int32):
    i = wp.tid()
    pair = i / points_per_pair
    point = i - pair * points_per_pair
    valid[i] = wp.int32(point == wp.int32(0) or (point < wp.int32(3) and (pair & wp.int32(15)) >= wp.int32(7)))


@wp.kernel(enable_backward=False, grid_stride=False)
def _compact_record(
    valid: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    src0: wp.array[wp.vec4],
    src1: wp.array[wp.vec4],
    src2: wp.array[wp.vec4],
    src3: wp.array[wp.vec4],
    src4: wp.array[wp.vec4],
    src5: wp.array[wp.vec4],
    dst0: wp.array[wp.vec4],
    dst1: wp.array[wp.vec4],
    dst2: wp.array[wp.vec4],
    dst3: wp.array[wp.vec4],
    dst4: wp.array[wp.vec4],
    dst5: wp.array[wp.vec4],
):
    i = wp.tid()
    if valid[i] == wp.int32(0):
        return
    output = offsets[i]
    dst0[output] = src0[i]
    dst1[output] = src1[i]
    dst2[output] = src2[i]
    dst3[output] = src3[i]
    dst4[output] = src4[i]
    dst5[output] = src5[i]


if __name__ == "__main__":
    main()
