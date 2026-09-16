# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Attribute Colibri kernel costs; eager profiling is not an FPS benchmark."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import warp as wp

from local_studies.colibri.check_public_slab_envelope import PublicSlabExample as Example
from local_studies.colibri.conservative_mesh_candidates import install_conservative_mesh_candidates
from newton.viewer import ViewerNull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-count", type=int, default=8)
    parser.add_argument("--mode", choices=("maximal", "reduced"), default="reduced")
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--forest", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--parallel-prepare", action="store_true")
    parser.add_argument("--mass-splitting", action="store_true")
    parser.add_argument("--contact-chunk-size", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--contact-offset-map", type=Path)
    parser.add_argument("--speculative-contact-gap-max", type=float, default=None)
    parser.add_argument("--max-colored-partitions", type=int, default=12)
    parser.add_argument("--mass-splitting-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/tmp/colibri_kernel_profile.json"))
    args = parser.parse_args()
    args.public_solver = True
    args.parallel_prepare = True
    args.mode = "maximal"
    args.no_graph = True
    args.outer_substeps = 2
    args.layout = "single_world"
    args.source_contact_offsets = True
    args.contact_gap = 0.001
    args.mesh_cylinders = True
    args.source_damping = False
    restore = install_conservative_mesh_candidates()
    try:
        example = Example(ViewerNull(), args)
    finally:
        restore()
    # Compile kernels and establish contact caches outside the measurement.
    for _ in range(args.warmup):
        example.step()
        example.test_post_step()
    with wp.ScopedTimer("Colibri eager frame", cuda_filter=wp.TIMING_KERNEL, print=False) as timer:
        example.step()
    grouped = defaultdict(lambda: [0, 0.0])
    for result in timer.timing_results:
        grouped[result.name][0] += 1
        grouped[result.name][1] += result.elapsed
    rows = [
        {"kernel": name, "count": count, "total_ms": elapsed}
        for name, (count, elapsed) in sorted(grouped.items(), key=lambda item: item[1][1], reverse=True)
    ]
    world = example.solver.world
    counts = world._copy_state.count_per_node.numpy()
    starts = world._partitioner.color_starts.numpy()
    colors = int(world._partitioner.num_colors.numpy()[0])
    overflow = 0
    if world.max_colored_partitions is not None and colors > world.max_colored_partitions:
        cap = world.max_colored_partitions
        overflow = int(starts[cap + 1] - starts[cap])
    report = {
        "copy_workload": {
            "slots_in_use": int(world._copy_state.highest_index_in_use.numpy()[0]),
            "max_copies_per_node": int(counts.max()) if counts.size else 0,
            "counts_per_node": counts.tolist(),
            "colors": colors,
            "overflow_constraints": overflow,
            "overflow_batches": (overflow + args.mass_splitting_batch_size - 1) // args.mass_splitting_batch_size,
        },
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "scope": "One eager frame, instrumented CUDA kernels; diagnostic costs only, not FPS or stability acceptance",
        "wall_ms": timer.elapsed,
        "kernel_ms_sum": sum(row["total_ms"] for row in rows),
        "kernels": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "kernels": rows[:15]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
