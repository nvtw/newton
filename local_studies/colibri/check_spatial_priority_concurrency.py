"""Local concurrent spatial-priority stress; no simulation or source changes."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    _export_and_reduce_contact_centered_two_spatial_depths,
)


@wp.func
def put(
    data: GlobalContactReducerData,
    points: wp.array[wp.vec3],
    gaps: wp.array[float],
    order: wp.array[int],
    index: int,
    margin: float,
):
    candidate = order[index]
    pair = candidate // 64
    point = points[candidate]
    _export_and_reduce_contact_centered_two_spatial_depths(
        2 * pair,
        2 * pair + 1,
        point,
        wp.vec3(0.0, 1.0, 0.0),
        gaps[candidate] + margin,
        candidate + 1,
        point,
        margin,
        margin + 16.0 / 65536.0,
        margin + 128.0 / 65536.0,
        point,
        wp.vec3(-1.0),
        wp.vec3(1.0),
        wp.vec3i(4),
        data,
        data.deterministic,
    )


@wp.kernel
def concurrent(
    data: GlobalContactReducerData,
    points: wp.array[wp.vec3],
    gaps: wp.array[float],
    order: wp.array[int],
    margin: float,
):
    put(data, points, gaps, order, wp.tid(), margin)


@wp.kernel
def serial(
    data: GlobalContactReducerData,
    points: wp.array[wp.vec3],
    gaps: wp.array[float],
    order: wp.array[int],
    margin: float,
):
    for i in range(order.shape[0]):
        put(data, points, gaps, order, i, margin)


def logical_slots(reducer, points, gaps, margin):
    """Decode every published slot and certify its live buffer ownership."""
    size = reducer.hashtable.capacity
    active = reducer.hashtable.active_slots.numpy()
    keys = reducer.hashtable.keys.numpy()
    values = reducer.ht_values.numpy()
    fingerprints = reducer.contact_fingerprints.numpy()
    geometry = reducer.position_depth.numpy()
    pairs = reducer.shape_pairs.numpy()
    count = int(reducer.contact_count.numpy()[0])
    assert count <= reducer.capacity
    assert int(reducer.ht_insert_failures.numpy()[0]) == 0
    mask = (1 << 20) - 1 if reducer.deterministic else 0xFFFFFFFF
    result = {}
    for entry in active[: active[size]]:
        for slot in range(reducer.values_per_key):
            packed = int(values[slot * size + entry])
            if not packed:
                continue
            cid = packed & mask
            assert 0 < cid <= count, (cid, count)
            candidate = int(fingerprints[cid]) - 1
            assert 0 <= candidate < len(points)
            pair = candidate // 64
            np.testing.assert_array_equal(pairs[cid], [2 * pair, 2 * pair + 1])
            np.testing.assert_array_equal(geometry[cid, :3], points[candidate])
            assert geometry[cid, 3] == np.float32(gaps[candidate] + margin)
            result[(int(keys[entry]), slot)] = candidate
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="/tmp/spatial_priority_concurrency_cpu.json")
    args = parser.parse_args()
    rng = np.random.default_rng(9291)
    count = 16 * 64
    points = rng.integers(-3000, 3001, (count, 3)).astype(np.float32) / 4096
    points[:, 1] = 0
    # Unique dyadic depths keep priority translation exact in FP32.
    gaps = np.tile(np.arange(1, 65), 16).astype(np.float32) / 65536
    for pair in range(1, 16, 2):
        gaps[64 * pair : 64 * pair + 3] *= -1
    arrays = [wp.array(points, dtype=wp.vec3, device=args.device), wp.array(gaps, dtype=float, device=args.device)]
    orders = [np.arange(count, dtype=np.int32), np.arange(count - 1, -1, -1, dtype=np.int32)]
    orders += [rng.permutation(count).astype(np.int32) for _ in range(4)]
    reports = []
    for deterministic in (False, True):
        for reclaim in (False, True):
            reducer = GlobalContactReducer(
                capacity=2 * count,
                device=args.device,
                deterministic=deterministic,
                enable_contact_reclamation=reclaim,
                hashtable_size_factor=1.0,
            )
            reference = None
            for margin in (0.0, 0.03125):
                for permutation, order in enumerate(orders):
                    for kernel in (serial, concurrent):
                        reducer.clear()
                        inputs = [
                            reducer.get_data_struct(),
                            *arrays,
                            wp.array(order, dtype=int, device=args.device),
                            margin,
                        ]
                        wp.launch(kernel, count if kernel == concurrent else 1, inputs=inputs, device=args.device)
                        slots = logical_slots(reducer, points, gaps, margin)
                        if reference is None:
                            reference = slots
                        assert slots == reference, (deterministic, reclaim, margin, permutation, kernel.key)
                        reports.append(
                            {
                                "deterministic": deterministic,
                                "reclaim": reclaim,
                                "margin": margin,
                                "permutation": permutation,
                                "kernel": kernel.key,
                                "slots": len(slots),
                            }
                        )
    result = {
        "status": "PASS",
        "device": args.device,
        "candidates": count,
        "cases": len(reports),
        "cases_detail": reports,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "cases_detail"}))


if __name__ == "__main__":
    main()
